"""
causal_pipeline/step3_causal_models.py
════════════════════════════════════════
Шаг 3: Каузальные методы прогнозирования

Метод 1: CausalImpact  — байесовская структурная модель временных рядов
                          (Google's causalimpact / tfcausalimpact)
Метод 2: DML           — Double Machine Learning (EconML)
                          residual-on-residual регрессия с RF/XGBoost
Метод 3: VAR+Granger   — Vector Autoregression с ограничениями
                          по значимости Granger-тестов
Метод 4: IV-2SLS       — Instrumental Variables (2-stage least squares)
                          для идентификации каузального эффекта

Каждый метод:
  - Обучается на train (80%)
  - Прогнозирует на test (20%)
  - Возвращает: predictions, metrics (RMSE, MAE, MAPE, R²), diagnostics
"""

import warnings
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATASETS_DIR = Path("datasets")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step1_dag_definitions import ALL_DAGS
from step2_variable_selection import (
    load_dataset, prepare_stationary, run_variable_selection
)

TRAIN_RATIO = 0.80

# ══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str = "") -> Dict[str, float]:
    """Вычисляет RMSE, MAE, MAPE, R²."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t, y_p = y_true[mask], y_pred[mask]
    if len(y_t) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan}
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae  = mean_absolute_error(y_t, y_p)
    mape = np.mean(np.abs((y_t - y_p) / (np.abs(y_t) + 1e-9))) * 100
    r2   = r2_score(y_t, y_p)
    metrics = {"RMSE": round(rmse, 6), "MAE": round(mae, 6),
               "MAPE": round(mape, 4), "R2": round(r2, 4)}
    if label:
        log.info(f"    {label}: RMSE={rmse:.4f}  MAE={mae:.4f}  "
                 f"MAPE={mape:.2f}%  R²={r2:.4f}")
    return metrics


def prepare_ml_data(df: pd.DataFrame, target: str,
                    controls: List[str],
                    causes: List[str],
                    horizon: int = 1,
                    n_lags: int = 5
                    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Формирует матрицу признаков X и целевой вектор y.
    Включает лаги целевого ряда + контрольные переменные.
    """
    feature_cols = list(dict.fromkeys(causes + controls))
    available = [c for c in feature_cols if c in df.columns]

    feat = pd.DataFrame(index=df.index)

    # Лаги целевого ряда
    for lag in range(1, n_lags + 1):
        feat[f"{target}_lag{lag}"] = df[target].shift(lag)

    # Лаги предикторов
    for col in available:
        for lag in range(0, 3):
            key = f"{col}_lag{lag}" if lag > 0 else col
            if key in df.columns:
                feat[key] = df[key]
            else:
                feat[f"{col}_lag{lag}"] = df[col].shift(lag)

    y = df[target].shift(-horizon)  # прогноз на horizon шагов вперёд
    combined = feat.join(y.rename("__target__")).dropna()
    X = combined.drop("__target__", axis=1)
    y_clean = combined["__target__"]
    return X, y_clean


def train_test_split_ts(X: pd.DataFrame, y: pd.Series,
                        ratio: float = TRAIN_RATIO):
    """Временной split без перемешивания."""
    n = len(X)
    cut = int(n * ratio)
    return (X.iloc[:cut], X.iloc[cut:],
            y.iloc[:cut], y.iloc[cut:])


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД 1: SCM-прогноз через DoWhy (Structural Causal Model)
# ══════════════════════════════════════════════════════════════════════════════

def run_scm_dowhy(df: pd.DataFrame, target: str,
                  causes: List[str],
                  controls: List[str],
                  dag: object = None,
                  horizon: int = 1) -> Dict:
    """
    SCM-прогноз (Structural Causal Model) через DoWhy + Ridge/GBM.

    Почему не CausalImpact
    ─────────────────────
    CausalImpact предназначен для оценки эффекта конкретного вмешательства
    (intervention): он строит контрфактик «что было бы без события» и меряет
    разницу. В пайплайне нет заданного вмешательства — есть задача прогноза.
    Применение CausalImpact к train/test split методологически некорректно:
    post_period — это не период вмешательства, а просто тестовая выборка,
    и инференс CausalImpact не равнозначен h-step ahead прогнозу.

    Что делает SCM-прогноз
    ──────────────────────
    1. DoWhy строит CausalModel из DAG (граф задаётся через causes/controls).
    2. Идентифицирует backdoor adjustment set — переменные Z, блокирующие
       все backdoor-пути от causes X к target Y.
    3. Опционально верифицирует идентификацию через рефутацию
       (placebo_treatment_refuter, random_common_cause).
    4. Строит прогнозную модель: Y(t+h) ~ f(X_causes_lag, Z_backdoor_lag),
       где набор регрессоров определён каузально, а не корреляционно.
       Используется RidgeCV (линейный, интерпретируемый) и GBM (нелинейный).
       Финальный прогноз — среднее взвешенное (ensemble).

    Прямой h-step прогноз
    ─────────────────────
    Для горизонта h > 1: целевая переменная Y сдвигается на -h при обучении
    (y[t+h] ~ X[t]), что соответствует прямому (DIRECT) методу мультишаговых
    прогнозов — в отличие от рекурсивного (накопление ошибки).

    Fallback
    ────────
    При отсутствии DoWhy или ошибке — откат к Ridge-регрессии с ручным
    backdoor adjustment (causes_lags + controls_lags).

    Параметры
    ---------
    dag      : AssetDAG объект из step1 (опционально, для DoWhy графа)
    horizon  : горизонт прогноза в шагах
    """
    log.info(f"  [SCM-DoWhy] {target}  h={horizon}")

    # ── Подготовка доступных переменных ──────────────────────────────────────
    avail_causes   = [c for c in causes   if c in df.columns]
    avail_controls = [c for c in controls if c in df.columns]

    if not avail_causes:
        avail_causes = [c for c in df.columns
                        if c != target and not c.startswith(f"{target}_")][:3]

    all_features = list(dict.fromkeys(avail_causes + avail_controls))
    if not all_features:
        return {"method": "SCM_SKIPPED", "metrics": {}, "predictions": {}}

    # ── Матрица признаков с лагами ────────────────────────────────────────────
    feat = pd.DataFrame(index=df.index)
    # Лаги целевого ряда (авторегрессионная компонента)
    for lag in range(1, 4):
        feat[f"{target}_lag{lag}"] = df[target].shift(lag)
    # Лаги причин (X) и контролей (Z)
    for col in all_features:
        for lag in range(1, 3):
            feat[f"{col}_lag{lag}"] = df[col].shift(lag)
    # Целевой вектор: прямой h-step прогноз
    feat["__y_h__"] = df[target].shift(-horizon)

    feat = feat.dropna()
    if len(feat) < 60:
        return {"method": "SCM_SKIPPED_insufficient_data", "metrics": {}}

    cut = int(len(feat) * TRAIN_RATIO)
    pred_cols = [c for c in feat.columns if c != "__y_h__"]
    X_all = feat[pred_cols].values
    y_all = feat["__y_h__"].values

    X_tr, X_te = X_all[:cut], X_all[cut:]
    y_tr, y_te = y_all[:cut], y_all[cut:]

    # ── Колонки по каузальным ролям ──────────────────────────────────────────
    cause_feat_cols   = [c for c in pred_cols
                         if any(c.startswith(f"{ca}_lag") for ca in avail_causes)]
    control_feat_cols = [c for c in pred_cols
                         if any(c.startswith(f"{co}_lag") for co in avail_controls)
                         and c not in cause_feat_cols]
    ar_cols           = [c for c in pred_cols if c.startswith(f"{target}_lag")]

    # Backdoor-корректный набор: causes + controls + AR (без лишних)
    backdoor_cols = list(dict.fromkeys(ar_cols + cause_feat_cols + control_feat_cols))
    if not backdoor_cols:
        backdoor_cols = pred_cols

    # ── DoWhy: идентификация и рефутация ─────────────────────────────────────
    dowhy_backdoor: Optional[List[str]] = None
    refutation_pval: Optional[float]   = None

    try:
        from dowhy import CausalModel

        # Строим граф на уровне исходных переменных (не лагов)
        if avail_causes and avail_controls:
            edges = (
                [f"{c} -> {target};" for c in avail_causes] +
                [f"{z} -> {target};" for z in avail_controls] +
                [f"{z} -> {c};" for z in avail_controls for c in avail_causes]
            )
            # graph_str не используется — pydot не требуется
            # graph_str = "digraph { " + " ".join(edges) + " }"

            # Подвыборка для DoWhy (работаем на исходном df, не на feat)
            dw_cols = list(dict.fromkeys([target] + avail_causes + avail_controls))
            dw_df   = df[dw_cols].dropna().iloc[:cut]  # только train

            if len(dw_df) >= 30 and avail_causes:
                dw_model = CausalModel(
                    data=dw_df,
                    treatment=avail_causes,
                    outcome=target,
                    common_causes=avail_controls if avail_controls else None,
                    # graph не передаётся — избегаем зависимости от pydot
                )
                identified = dw_model.identify_effect(
                    proceed_when_unidentifiable=True
                )
                dowhy_backdoor = identified.get_backdoor_variables()
                log.info(f"    DoWhy backdoor set: {dowhy_backdoor}")

                # Рефутация: placebo treatment на первом причинном ряде
                # DoWhy placebo_treatment_refuter требует 1D treatment —
                # при множественном treatment берём только первый для теста
                try:
                    first_cause  = avail_causes[:1]
                    dw_model_1t  = CausalModel(
                        data=dw_df,
                        treatment=first_cause,
                        outcome=target,
                        common_causes=avail_controls if avail_controls else None,
                    )
                    identified_1t = dw_model_1t.identify_effect(
                        proceed_when_unidentifiable=True
                    )
                    dw_est = dw_model_1t.estimate_effect(
                        identified_1t,
                        method_name="backdoor.linear_regression",
                        control_value=0, treatment_value=1,
                    )
                    ref = dw_model_1t.refute_estimate(
                        identified_1t, dw_est,
                        method_name="placebo_treatment_refuter",
                        placebo_type="permute",
                        num_simulations=50,
                    )
                    refutation_pval = ref.refutation_result.get("p_value")
                    log.info(f"    Рефутация ({first_cause[0]}) p={refutation_pval:.3f} "
                             f"({'OK' if refutation_pval and refutation_pval > 0.05 else 'WARN: возможна spurious причинность'})")
                except Exception as re:
                    log.warning(f"    Рефутация пропущена: {re}")

                # Если DoWhy нашёл непустой backdoor — уточняем набор контролей
                if dowhy_backdoor:
                    dw_ctrl_lag = [c for c in pred_cols
                                   if any(c.startswith(f"{z}_lag")
                                          for z in dowhy_backdoor)]
                    if dw_ctrl_lag:
                        control_feat_cols = dw_ctrl_lag
                        backdoor_cols = list(dict.fromkeys(
                            ar_cols + cause_feat_cols + control_feat_cols
                        ))
                        log.info(f"    Обновлён backdoor feature set: "
                                 f"{len(backdoor_cols)} колонок")

    except ImportError:
        log.warning("    dowhy не установлен — используем ручной backdoor")
    except Exception as e:
        log.warning(f"    DoWhy ошибка: {e}")

    # ── Прогнозные модели на backdoor-корректном наборе признаков ─────────────
    scaler = StandardScaler()
    bd_idx = [pred_cols.index(c) for c in backdoor_cols if c in pred_cols]

    if not bd_idx:
        bd_idx = list(range(len(pred_cols)))

    X_bd_tr = scaler.fit_transform(X_tr[:, bd_idx])
    X_bd_te = scaler.transform(X_te[:, bd_idx])

    preds_list = []

    # Модель 1: RidgeCV (линейная, интерпретируемая)
    try:
        from sklearn.linear_model import RidgeCV
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        ridge.fit(X_bd_tr, y_tr)
        preds_list.append(("Ridge", ridge.predict(X_bd_te), 0.4))
        log.info(f"    Ridge alpha={ridge.alpha_:.3f}")
    except Exception as e:
        log.warning(f"    Ridge ошибка: {e}")

    # Модель 2: GradientBoosting (нелинейная)
    try:
        gbm = GradientBoostingRegressor(
            n_estimators=150, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            random_state=42
        )
        gbm.fit(X_bd_tr, y_tr)
        preds_list.append(("GBM", gbm.predict(X_bd_te), 0.6))
        log.info(f"    GBM n_features={X_bd_tr.shape[1]}")
    except Exception as e:
        log.warning(f"    GBM ошибка: {e}")

    if not preds_list:
        log.error(f"    Все модели SCM упали")
        return {"method": "SCM_FAILED", "metrics": {}, "predictions": {}}

    # Взвешенное среднее (ensemble)
    total_w  = sum(w for _, _, w in preds_list)
    y_pred   = sum(pred * (w / total_w) for _, pred, w in preds_list)
    model_names = "+".join(name for name, _, _ in preds_list)

    metrics = compute_metrics(y_te, y_pred, f"SCM_{model_names}(h={horizon})")

    return {
        "method":            f"SCM_{model_names}",
        "metrics":           metrics,
        "predictions":       dict(zip(
            feat.index[cut:].strftime("%Y-%m-%d").tolist(),
            y_pred.tolist()
        )),
        "dowhy_backdoor":    dowhy_backdoor,
        "refutation_pval":   refutation_pval,
        "backdoor_n_features": len(bd_idx),
    }


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД 2: Double Machine Learning (DML)
# ══════════════════════════════════════════════════════════════════════════════

def run_dml(df: pd.DataFrame, target: str,
            causes: List[str],
            controls: List[str],
            horizon: int = 1) -> Dict:
    """
    Double Machine Learning (Chernozhukov et al., 2018):
      1. Partialling out: предсказываем Y и каждый X_j из Z (controls)
      2. Residual-on-residual регрессия: ε_Y ~ ε_X
      3. Прогноз = θ̂ * X + f(Z)

    Использует EconML или ручную реализацию с cross-fitting.
    """
    log.info(f"  [DML] {target}")

    X_mat, y_vec = prepare_ml_data(df, target, controls, causes,
                                   horizon=horizon, n_lags=5)
    if len(X_mat) < 60:
        return {"method": "DML_SKIPPED", "metrics": {}}

    X_tr, X_te, y_tr, y_te = train_test_split_ts(X_mat, y_vec)

    # ── Поиск колонок причин (treatment) в матрице признаков ─────────────────
    # prepare_ml_data создаёт: col (lag=0, без суффикса), col_lag1, col_lag2
    all_cols = X_mat.columns.tolist()
    cause_cols = []
    for base in causes:
        # Приоритет: точное имя → col_lag1 (ближайший лаг) → startswith
        for cand in [base, f"{base}_lag1",
                     next((c for c in all_cols if c.startswith(base + "_")), None)]:
            if cand and cand in all_cols and cand not in cause_cols:
                cause_cols.append(cand)
                break
    cause_cols = cause_cols[:min(5, len(all_cols) // 2)]  # не более 5 и не больше половины
    control_cols = [c for c in all_cols if c not in cause_cols]

    if not cause_cols or not control_cols:
        # Последний fallback: делим матрицу пополам
        mid = max(1, len(all_cols) // 2)
        cause_cols, control_cols = all_cols[:mid], all_cols[mid:]
        log.warning(f"    DML: fallback split ({len(cause_cols)} treat / {len(control_cols)} ctrl)")

    # Попытка использовать EconML LinearDML
    try:
        from econml.dml import LinearDML
        from sklearn.linear_model import LassoCV

        est = LinearDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
            model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3),
            discrete_treatment=False,
            cv=3,
            random_state=42
        )
        T_tr = X_tr[cause_cols].values
        W_tr = X_tr[control_cols].values
        T_te = X_te[cause_cols].values
        W_te = X_te[control_cols].values

        est.fit(y_tr.values.ravel(), T=T_tr, X=None, W=W_tr)
        # EconML prediction: causal effect * T
        effects = est.effect(T0=np.zeros_like(T_te), T1=T_te, X=None)
        # Baseline от control: отдельная регрессия
        rf_base = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_base.fit(W_tr, y_tr.values.ravel() -
                    est.effect(T0=np.zeros_like(T_tr), T1=T_tr).flatten())
        baseline_pred = rf_base.predict(W_te)
        y_pred = baseline_pred + effects.flatten()

        metrics = compute_metrics(y_te.values.ravel(), y_pred, "DML_EconML")
        return {
            "method":    "DML_EconML",
            "metrics":   metrics,
            "predictions": dict(zip(
                y_te.index.strftime("%Y-%m-%d").tolist(),
                y_pred.tolist()
            )),
            "causal_effects": effects.mean(0).tolist(),
        }
    except (ImportError, Exception) as e:
        log.warning(f"    EconML DML: {e}. Использую ручной DML.")

    # Ручная реализация DML (cross-fitting, 2 фолда)
    return _manual_dml(X_tr, X_te, y_tr, y_te, cause_cols, control_cols)


def _manual_dml(X_tr, X_te, y_tr, y_te,
                cause_cols: List[str],
                control_cols: List[str]) -> Dict:
    """
    Ручной DML:
      Step 1: ε_Y = Y - E[Y|Z]  (partialling out Y)
      Step 2: ε_T = T - E[T|Z]  (partialling out each treatment)
      Step 3: OLS ε_Y ~ ε_T     (causal coefficient θ)
      Step 4: Forecast = θ*T_test + RF(Z_test)
    """
    from sklearn.linear_model import LinearRegression, RidgeCV

    W_tr = X_tr[control_cols].values
    W_te = X_te[control_cols].values
    T_tr = X_tr[cause_cols].values
    T_te = X_te[cause_cols].values

    # Step 1: partialling out Y
    m_y = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=0)
    m_y.fit(W_tr, y_tr.values.ravel())
    res_y_tr = y_tr.values.ravel() - m_y.predict(W_tr)

    # Step 2: partialling out T (каждый treatment по отдельности)
    res_T_tr = np.zeros_like(T_tr, dtype=float)
    m_t_list = []
    for j in range(T_tr.shape[1]):
        m_t = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=j)
        m_t.fit(W_tr, T_tr[:, j])
        res_T_tr[:, j] = T_tr[:, j] - m_t.predict(W_tr)
        m_t_list.append(m_t)

    # Step 3: OLS residual-on-residual
    theta_model = LinearRegression(fit_intercept=False)
    theta_model.fit(res_T_tr, res_y_tr)
    theta = theta_model.coef_

    # Step 4: Prediction
    baseline = m_y.predict(W_te)
    res_T_te = T_te - np.column_stack(
        [m_t.predict(W_te) for m_t in m_t_list]
    )
    y_pred = baseline + res_T_te @ theta

    metrics = compute_metrics(y_te.values, y_pred, "DML_manual")
    return {
        "method":    "DML_manual",
        "metrics":   metrics,
        "predictions": dict(zip(
            y_te.index.strftime("%Y-%m-%d").tolist(),
            y_pred.tolist()
        )),
        "theta": dict(zip(cause_cols, theta.tolist())),
    }


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД 3: VAR с ограничениями Granger
# ══════════════════════════════════════════════════════════════════════════════

def run_var_granger(df: pd.DataFrame, target: str,
                    causes: List[str],
                    controls: List[str],
                    max_lags: int = 6,
                    horizon: int = 1) -> Dict:
    """
    VAR + Granger causality:
      1. Тест Гренджера для каждого кандидата X → Y
      2. Оставляем только статистически значимые (p < 0.05)
      3. Подбираем лаг VAR по AIC/BIC
      4. Прогноз на горизонт horizon
    """
    log.info(f"  [VAR+Granger] {target}")

    import statsmodels.api as sm
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.stattools import grangercausalitytests

    candidates = list(dict.fromkeys(causes + controls))
    available  = [c for c in candidates if c in df.columns]

    sub = df[[target] + available].dropna()
    n   = len(sub)
    if n < 50:
        return {"method": "VAR_SKIPPED", "metrics": {}}

    cut = int(n * TRAIN_RATIO)
    train = sub.iloc[:cut]
    test  = sub.iloc[cut:]

    # ── Granger тест ──────────────────────────────────────────────────────────
    significant_vars = []
    granger_pvalues  = {}
    for col in available:
        try:
            pair = train[[target, col]].dropna()
            if len(pair) < 30:
                continue
            res = grangercausalitytests(pair, maxlag=min(max_lags, 4),
                                        verbose=False)
            # Минимальный p-value по всем лагам (F-test)
            p_vals = [res[lag][0]["ssr_ftest"][1]
                      for lag in range(1, min(max_lags, 4) + 1)]
            min_p = min(p_vals)
            granger_pvalues[col] = round(min_p, 4)
            if min_p < 0.05:
                significant_vars.append(col)
        except Exception:
            pass

    log.info(f"    Granger значимых: {len(significant_vars)}/{len(available)}")
    log.info(f"    {dict(list(granger_pvalues.items())[:5])}")

    # Если ничего не прошло — берём топ-3 по p-value
    if not significant_vars:
        significant_vars = sorted(granger_pvalues, key=granger_pvalues.get)[:3]
        log.info(f"    Используем топ-3 по p-value: {significant_vars}")

    # ── Подбор лага VAR по AIC ────────────────────────────────────────────────
    var_cols = [target] + significant_vars[:5]  # VAR до 6 переменных
    var_data = train[var_cols].dropna()

    try:
        model = VAR(var_data)
        lag_order = model.select_order(maxlags=max_lags)
        best_lag  = lag_order.aic
        if best_lag == 0:
            best_lag = 1
    except Exception:
        best_lag = 2

    # ── Обучение и прогноз ───────────────────────────────────────────────────
    try:
        fitted = VAR(var_data).fit(maxlags=best_lag, ic=None)

        # Прогнозируем шаг за шагом (rolling forecast)
        history  = var_data.values.copy()
        preds    = []
        test_var = test[var_cols].dropna()

        for i in range(len(test_var)):
            fc = fitted.forecast(history[-best_lag:], steps=horizon)
            preds.append(fc[horizon - 1, 0])  # индекс 0 = target
            # Обновляем историю реальными данными
            history = np.vstack([history, test_var.iloc[i].values])

        preds = np.array(preds)
        y_true = test_var[target].values[:len(preds)]
        metrics = compute_metrics(y_true, preds, f"VAR(lag={best_lag})")

        return {
            "method":  f"VAR_Granger(lag={best_lag})",
            "metrics": metrics,
            "predictions": dict(zip(
                test_var.index[:len(preds)].strftime("%Y-%m-%d").tolist(),
                preds.tolist()
            )),
            "granger_pvalues":  granger_pvalues,
            "significant_vars": significant_vars,
            "best_lag": best_lag,
        }
    except Exception as e:
        log.error(f"    VAR ошибка: {e}")
        return {"method": "VAR_FAILED", "metrics": {}}


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД 4: IV-2SLS (инструментальные переменные)
# ══════════════════════════════════════════════════════════════════════════════

def run_iv_2sls(df: pd.DataFrame, target: str,
                causes: List[str],
                controls: List[str],
                instruments: List[str],
                horizon: int = 1) -> Dict:
    """
    Two-Stage Least Squares:
      Stage 1: X̂ = Z + controls  (инструменты предсказывают причину)
      Stage 2: Y = X̂ + controls  (используем X̂ вместо X)

    Устраняет endogeneity (simultaneity bias).
    """
    log.info(f"  [IV-2SLS] {target}")

    try:
        from linearmodels.iv import IV2SLS as lmIV
        use_linearmodels = True
    except ImportError:
        use_linearmodels = False

    # Формируем данные
    X_mat, y_vec = prepare_ml_data(df, target, controls,
                                   causes + instruments,
                                   horizon=horizon, n_lags=3)
    if len(X_mat) < 60:
        return {"method": "IV_SKIPPED", "metrics": {}}

    X_tr, X_te, y_tr, y_te = train_test_split_ts(X_mat, y_vec)

    # Находим доступные инструменты и причины в матрице
    all_iv_cols = X_mat.columns.tolist()

    def _find_best(base_names, col_pool, max_n=5):
        """Ищет представителей base_names в col_pool: точное имя → _lag1 → startswith."""
        found = []
        for base in base_names:
            for cand in [base, f"{base}_lag1",
                         next((c for c in col_pool if c.startswith(base + "_")), None)]:
                if cand and cand in col_pool and cand not in found:
                    found.append(cand)
                    break
        return found[:max_n]

    inst_cols  = _find_best(instruments, all_iv_cols, max_n=3)
    endog_cols = _find_best(causes,      all_iv_cols, max_n=5)
    exog_cols  = [c for c in X_mat.columns
                  if c not in inst_cols + endog_cols][:8]

    if not inst_cols or not endog_cols:
        log.warning(f"    2SLS: нет инструментов или эндогенных переменных")
        return _manual_2sls(X_tr, X_te, y_tr, y_te,
                            endog_cols or X_mat.columns[:2].tolist(),
                            exog_cols)

    if use_linearmodels:
        try:
            import statsmodels.api as sm
            # linearmodels ожидает формат: endog, exog, dependent, instruments
            train_df = pd.concat([y_tr.rename("__y__"), X_tr], axis=1).dropna()
            endog = train_df[endog_cols]
            exog  = sm.add_constant(train_df[exog_cols])
            instruments_df = train_df[inst_cols]

            model = lmIV(train_df["__y__"], exog, endog, instruments_df)
            res   = model.fit(cov_type="robust")

            # Прогноз: вычисляем вручную через коэффициенты
            test_df = pd.concat([y_te.rename("__y__"), X_te], axis=1).dropna()
            X_test_exog = sm.add_constant(test_df[exog_cols])
            X_test_endog = test_df[endog_cols]

            coef_exog  = res.params[[c for c in res.params.index
                                     if c in X_test_exog.columns]]
            coef_endog = res.params[[c for c in res.params.index
                                     if c in X_test_endog.columns]]

            pred = (X_test_exog[coef_exog.index].values @ coef_exog.values +
                    X_test_endog[coef_endog.index].values @ coef_endog.values)

            metrics = compute_metrics(test_df["__y__"].values, pred, "IV-2SLS")
            return {
                "method":  "IV_2SLS",
                "metrics": metrics,
                "predictions": dict(zip(
                    test_df.index.strftime("%Y-%m-%d").tolist(),
                    pred.tolist()
                )),
                "first_stage_f": float(res.first_stage.diagnostics["f.stat"].mean())
                                  if hasattr(res, "first_stage") else None,
            }
        except Exception as e:
            log.warning(f"    linearmodels 2SLS ошибка: {e}")

    return _manual_2sls(X_tr, X_te, y_tr, y_te, endog_cols, exog_cols,
                        inst_cols)


def _manual_2sls(X_tr, X_te, y_tr, y_te,
                 endog_cols, exog_cols,
                 inst_cols: Optional[List[str]] = None) -> Dict:
    """Ручная реализация 2SLS через OLS."""
    from sklearn.linear_model import LinearRegression

    if not endog_cols:
        endog_cols = X_tr.columns[:1].tolist()

    W_tr  = X_tr[exog_cols].values if exog_cols else np.ones((len(X_tr), 1))
    W_te  = X_te[exog_cols].values if exog_cols else np.ones((len(X_te), 1))
    T_tr  = X_tr[endog_cols].values
    T_te  = X_te[endog_cols].values

    if inst_cols:
        Z_tr = X_tr[inst_cols].values
        # Stage 1
        stage1 = LinearRegression()
        Z_W_tr = np.hstack([Z_tr, W_tr])
        stage1.fit(Z_W_tr, T_tr)
        T_hat_tr = stage1.predict(Z_W_tr)

        Z_te = X_te[inst_cols].values
        Z_W_te = np.hstack([Z_te, W_te])
        T_hat_te = stage1.predict(Z_W_te)
    else:
        T_hat_tr = T_tr
        T_hat_te = T_te

    # Stage 2
    stage2 = LinearRegression()
    stage2.fit(np.hstack([T_hat_tr, W_tr]), y_tr.values)
    y_pred = stage2.predict(np.hstack([T_hat_te, W_te]))

    metrics = compute_metrics(y_te.values, y_pred, "2SLS_manual")
    return {
        "method":  "2SLS_manual",
        "metrics": metrics,
        "predictions": dict(zip(
            y_te.index.strftime("%Y-%m-%d").tolist(),
            y_pred.tolist()
        )),
    }


# ══════════════════════════════════════════════════════════════════════════════
# АДАПТИВНЫЙ ГОРИЗОНТ ПРОГНОЗА
# ══════════════════════════════════════════════════════════════════════════════

# Горизонты прогноза по частоте данных
HORIZONS = {
    "D":  [1, 5, 21],   # дневные:  1 день, 1 неделя, 1 месяц
    "B":  [1, 5, 21],   # рабочие дни — то же
    "ME": [1, 3, 6],    # месячные: 1 мес, квартал, полгода
    "MS": [1, 3, 6],
    "M":  [1, 3, 6],
    "W":  [1, 4, 13],   # недельные: 1 нед, 1 мес, квартал
}

def get_horizons(freq: str) -> List[int]:
    return HORIZONS.get(freq, [1, 5, 21])


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК ОДНОГО АКТИВА С ТРЕМЯ НАБОРАМИ КОНТРОЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_control_set(df_stat: pd.DataFrame,
                         target: str,
                         causes: List[str],
                         controls: List[str],
                         instruments: List[str],
                         horizons: List[int],
                         label: str) -> Dict:
    """
    Запускает все 4 каузальных метода для одного набора контролей Z
    и по всем горизонтам прогноза.
    Возвращает: {horizon: {method: result}}
    """
    result_by_horizon = {}
    for h in horizons:
        log.info(f"    horizon={h} | controls={label}")
        h_res = {}

        h_res["causal_impact"] = run_scm_dowhy(
            df_stat, target, causes, controls,
            dag=ALL_DAGS.get(target), horizon=h
        )

        h_res["dml"] = run_dml(df_stat, target, causes, controls,
                               horizon=h)

        h_res["var_granger"] = run_var_granger(df_stat, target, causes,
                                               controls, horizon=h)

        h_res["iv_2sls"] = run_iv_2sls(df_stat, target, causes, controls,
                                       instruments, horizon=h)

        result_by_horizon[f"h{h}"] = h_res

    return result_by_horizon


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК ВСЕХ КАУЗАЛЬНЫХ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════

def run_all_causal_models(var_selection: Dict) -> Dict:
    """
    Для каждого из 10 активов запускает 4 каузальных метода ×
    3 варианта контролей (heuristic / causal_expert / causal_pc) ×
    N горизонтов прогноза (адаптивно по частоте данных).

    Структура результата:
      {target: {
          "controls_heuristic":     {h1: {method: result}, ...},
          "controls_causal_expert": {...},
          "controls_causal_pc":     {...},
          "freq": "D" | "ME",
          "horizons": [1, 5, 21]
       }}
    """
    all_results = {}

    for target, info in var_selection.items():
        dag = ALL_DAGS.get(target)
        if dag is None:
            continue

        log.info(f"\n{'═'*60}")
        log.info(f"  КАУЗАЛЬНЫЕ МОДЕЛИ: {dag.name} ({target})  "
                 f"[freq={dag.freq}]")
        log.info(f"{'═'*60}")

        df_raw  = load_dataset(dag)
        if df_raw.empty or target not in df_raw.columns:
            log.warning(f"  Пропускаем {target}: нет данных")
            continue
        df_stat = prepare_stationary(df_raw, target)

        causes    = info.get("causes",      dag.causes)
        instrs    = info.get("instruments", dag.instruments)
        horizons  = get_horizons(dag.freq)

        # Три варианта наборов контролей
        control_sets = {
            "controls_heuristic":     info.get("heuristic",     []),
            "controls_causal_expert": info.get("causal_expert", []),
            "controls_causal_pc":     info.get("causal_pc",
                                               info.get("causal_expert", [])),
        }

        target_results = {
            "freq":    dag.freq,
            "horizons": horizons,
            "asset":   dag.name,
        }

        for ctrl_label, ctrl_vars in control_sets.items():
            log.info(f"\n  ▸ {ctrl_label}: {ctrl_vars}")
            target_results[ctrl_label] = _run_one_control_set(
                df_stat, target,
                causes, ctrl_vars, instrs,
                horizons, label=ctrl_label
            )

        all_results[target] = target_results

    # ── сериализация ──────────────────────────────────────────────────────────
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    out_path = RESULTS_DIR / "step3_causal_models.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_clean(all_results), f, ensure_ascii=False, indent=2)
    log.info(f"\n✓ Результаты step3 сохранены: {out_path}")
    return all_results


if __name__ == "__main__":
    log.info("Загружаем отбор переменных (step2)...")
    sel_path = RESULTS_DIR / "step2_variable_selection.json"
    if sel_path.exists():
        with open(sel_path) as f:
            var_sel = json.load(f)
    else:
        log.info("Запускаем step2...")
        var_sel = run_variable_selection(use_pcmci=True)

    run_all_causal_models(var_sel)
