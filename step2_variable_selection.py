"""
causal_pipeline/step2_variable_selection.py
════════════════════════════════════════════
Шаг 2: Отбор контрольных переменных Z тремя методами:

  (A) HEURISTIC — двухэтапный автоматический отбор без фиксированного top-k:
        Этап 1. LassoCV: отбирает нелинейно-скоррелированные предикторы
                путём L1-регуляризации с подбором alpha по кросс-валидации.
        Этап 2. Порог накопленной объяснённой дисперсии (PCA 80%):
                из кандидатов после LASSO оставляем минимальный набор,
                объясняющий >= variance_threshold дисперсии X-матрицы.
        Итог: множество Z без заранее заданного k — размер определяется данными.

  (B) EXPERT — экспертный backdoor criterion из DAG (step1):
        backdoor_set = confounders \\ mediators

  (C) PCMCI+ — автоматический каузальный отбор (tigramite):
        PCMCI+ (Runge et al., 2019) — алгоритм обнаружения причинных
        связей в временных рядах с учётом лагов и ориентации рёбер.
        Шаг 1. Skeleton: условные тесты независимости (ParCorr / RobustParCorr)
                убирают ложные рёбра, управляя FDR.
        Шаг 2. Orientation: v-структуры и правила Мо-Пирла ориентируют рёбра.
        Извлекаем: переменные, у которых есть направленное ребро X(t-tau) -> Y(t)
        хотя бы на одном лаге tau in [1, tau_max]. Исключаем медиаторы.
        Fallback при ошибке -> expert DAG.

Выходы (results/step2_variable_selection.json):
  heuristic      : Z из LassoCV + PCA
  causal_expert  : Z из экспертного DAG
  causal_pc      : Z из PCMCI+
  causal_union   : объединение expert + pcmci (приоритет эксперта)
  selection_meta : диагностики (lasso_alpha, n_pca_components, pcmci_parents, ...)
"""

from __future__ import annotations

import json
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step1_dag_definitions import ALL_DAGS, AssetDAG

DATASETS_DIR = Path("datasets")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ ЗАГРУЗКИ И ПРЕДОБРАБОТКИ
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(dag: AssetDAG, max_cols: int = 50) -> pd.DataFrame:
    """
    Загружает CSV-датасет актива и выполняет базовую фильтрацию:
      - оставляет только числовые колонки
      - удаляет колонки с >50% пропусков и нулевой дисперсией
      - при >max_cols столбцов оставляет целевой + топ-(max_cols-1) по корреляции
    При отсутствии файла возвращает синтетические данные для тестирования.
    """
    path = DATASETS_DIR / dag.file
    if not path.exists():
        log.warning(f"  Файл не найден: {path} — используем синтетические данные")
        return _synthetic_data(dag)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.select_dtypes(include=[np.number])
    df = df.dropna(thresh=int(len(df) * 0.5), axis=1)
    df = df.loc[:, df.std() > 1e-9]
    if len(df.columns) > max_cols and dag.target in df.columns:
        corrs = df.corr()[dag.target].abs().sort_values(ascending=False)
        keep  = [dag.target] + corrs.index[1:max_cols].tolist()
        df    = df[keep]
    return df


def _synthetic_data(dag: AssetDAG, n: int = 500) -> pd.DataFrame:
    """
    Синтетические данные с легкой каузальной структурой:
      Y[t] = sum 0.3*X_i[t] + шум
    Используется только при отсутствии реальных CSV-файлов.
    """
    np.random.seed(42)
    all_vars = list(dict.fromkeys(
        [dag.target] + dag.causes + dag.confounders + dag.instruments
    ))
    data = np.random.randn(n, len(all_vars))
    for i in range(1, len(all_vars)):
        data[:, 0] += 0.3 * data[:, i] + 0.1 * np.random.randn(n)
    freq = "ME" if dag.freq == "ME" else "B"
    idx  = pd.date_range("2010-01-04", periods=n, freq=freq)
    return pd.DataFrame(data, columns=all_vars, index=idx)


def prepare_stationary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Приводит все колонки к стационарности:
      - ADF-тест с автоматическим выбором лагов по AIC
      - p-value > 0.05  =>  берём первые разности
      - p-value <= 0.05 =>  оставляем как есть
    После взятия разностей удаляем первую строку (NaN).
    """
    from statsmodels.tsa.stattools import adfuller
    result = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 20:
            continue
        try:
            p_val = adfuller(s, autolag="AIC")[1]
        except Exception:
            p_val = 1.0
        result[col] = df[col].diff() if p_val > 0.05 else df[col]
    return pd.DataFrame(result, index=df.index).dropna()


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД A: HEURISTIC — корреляционный предфильтр + LassoCV
# ══════════════════════════════════════════════════════════════════════════════

def select_heuristic(
    df: pd.DataFrame,
    target: str,
    lasso_cv: int = 5,
    min_corr: float = 0.10,
    max_vars: int = 15,
) -> Tuple[List[str], Dict]:
    """
    Двухэтапный автоматический отбор без фиксированного top-k.

    Параметры
    ----------
    df       : стационарный датафрейм
    target   : имя целевого ряда
    lasso_cv : число фолдов в LassoCV
    min_corr : минимальный порог корреляции для предфильтра
    max_vars : жёсткий потолок числа итоговых переменных (страховка)

    Этап 1 — Предфильтр по корреляции (Pearson + Spearman):
      Убираем переменные с combined_corr < min_corr.
      Цель — сократить пространство кандидатов для LASSO, отбросив чистый шум.
      Не претендует на финальный отбор: это только убыстрение и стабилизация.

    Этап 2 — LassoCV:
      L1-регуляризация с подбором alpha по кросс-валидации (cv фолдов).
      Количество ненулевых коэффициентов определяется данными, а не параметром.
      Переменные сортируются по убыванию |coef|: наиболее важные для Y — первыми.
      Если LassoCV обнуляет все коэффициенты — ослабляем alpha x0.1.
      Fallback при любой ошибке — корреляционный топ-5.

    Возвращает: (список переменных, метаданные диагностики)
    """
    if target not in df.columns:
        log.error(f"  [{target}] целевой ряд не найден")
        return [], {}

    candidates = [
        c for c in df.columns
        if c != target and not c.startswith(f"{target}_")
    ]
    if len(candidates) < 2:
        return candidates, {"method": "fallback_no_candidates"}

    y = df[target].dropna()

    # Этап 1: предфильтр по корреляции
    corr_scores: Dict[str, float] = {}
    for col in candidates:
        x = df[col].dropna()
        idx = y.index.intersection(x.index)
        if len(idx) < 30:
            continue
        xi, yi = x[idx].values, y[idx].values
        try:
            p = abs(float(np.corrcoef(xi, yi)[0, 1]))
            s = abs(float(stats.spearmanr(xi, yi).correlation))
            if not (np.isnan(p) or np.isnan(s)):
                corr_scores[col] = 0.5 * p + 0.5 * s
        except Exception:
            pass

    pre_candidates = [c for c, v in corr_scores.items() if v >= min_corr]
    log.info(f"  [{target}] Предфильтр: {len(pre_candidates)}/{len(candidates)} "
             f"кандидатов (corr >= {min_corr})")

    if len(pre_candidates) < 2:
        pre_candidates = sorted(
            corr_scores, key=corr_scores.get, reverse=True  # type: ignore
        )[:5]
        log.info(f"  [{target}] Смягчаем предфильтр — берём топ-5")

    # Этап 2: LassoCV
    sub = df[[target] + pre_candidates].dropna()
    lasso_alpha: Optional[float] = None
    selected: List[str] = []

    if len(sub) < 40:
        log.warning(f"  [{target}] n={len(sub)} < 40 — LASSO пропускаем, "
                    f"используем топ-5 по корреляции")
        selected = sorted(
            corr_scores, key=corr_scores.get, reverse=True  # type: ignore
        )[:min(5, max_vars)]
    else:
        scaler = StandardScaler()
        X_sc  = scaler.fit_transform(sub[pre_candidates].values)
        y_arr = sub[target].values

        try:
            lasso_cv_obj = LassoCV(
                cv=lasso_cv, max_iter=10_000, random_state=42, n_jobs=-1
            )
            lasso_cv_obj.fit(X_sc, y_arr)
            lasso_alpha = float(lasso_cv_obj.alpha_)
            coefs       = lasso_cv_obj.coef_

            # Сортируем по убыванию |coef| — важнейшие для Y идут первыми
            pairs = [
                (pre_candidates[i], abs(float(c)))
                for i, c in enumerate(coefs) if abs(c) > 1e-6
            ]
            pairs.sort(key=lambda kv: kv[1], reverse=True)
            selected = [v for v, _ in pairs]

            if not selected:
                log.warning(f"  [{target}] LassoCV обнулил все коэфф. "
                             f"(alpha={lasso_alpha:.5f}) — ослабляем x0.1")
                lasso_soft = Lasso(alpha=lasso_alpha * 0.1, max_iter=10_000)
                lasso_soft.fit(X_sc, y_arr)
                pairs_soft = [
                    (pre_candidates[i], abs(float(c)))
                    for i, c in enumerate(lasso_soft.coef_) if abs(c) > 1e-6
                ]
                pairs_soft.sort(key=lambda kv: kv[1], reverse=True)
                selected = [v for v, _ in pairs_soft]

            log.info(f"  [{target}] LassoCV alpha={lasso_alpha:.5f}: "
                     f"{len(selected)} переменных")

        except Exception as e:
            log.warning(f"  [{target}] LassoCV ошибка: {e} — топ-5 по корреляции")
            selected = sorted(
                corr_scores, key=corr_scores.get, reverse=True  # type: ignore
            )[:min(5, max_vars)]

    if not selected:
        selected = sorted(
            corr_scores, key=corr_scores.get, reverse=True  # type: ignore
        )[:min(5, len(pre_candidates))]

    selected = selected[:max_vars]

    meta = {
        "method":           "LassoCV",
        "n_pre_candidates": len(pre_candidates),
        "n_selected":       len(selected),
        "lasso_alpha":      lasso_alpha,
        "n_final":          len(selected),
    }
    log.info(f"  [{target}] Heuristic итог ({len(selected)}): {selected}")
    return selected, meta


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД B: EXPERT — backdoor criterion из экспертного DAG
# ══════════════════════════════════════════════════════════════════════════════

def select_causal_expert(dag: AssetDAG) -> List[str]:
    """
    Экспертный backdoor adjustment set:
      backdoor_set = confounders \\ mediators

    Медиаторы исключаются намеренно: контроль переменной на каузальном пути
    X -> M -> Y блокирует сам измеряемый эффект (over-control bias).
    """
    backdoor_set = [z for z in dag.confounders if z not in dag.mediators]
    log.info(f"  [{dag.target}] Expert backdoor: {backdoor_set}")
    return backdoor_set


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД C: PCMCI+ — автоматическое обнаружение каузальных связей в ВР
# ══════════════════════════════════════════════════════════════════════════════

def select_causal_pcmci(
    df: pd.DataFrame,
    target: str,
    dag: AssetDAG,
    tau_max: int = 4,
    pc_alpha: float = 0.05,
    use_robust: bool = False,
) -> Tuple[List[str], Dict]:
    """
    Отбор переменных через PCMCI+ (tigramite).

    PCMCI+ (Runge et al., 2019, Science Advances):
      Расширение PC-алгоритма для временных рядов. Учитывает лаговую
      структуру зависимостей и ориентирует как лаговые X(t-tau)->Y(t),
      так и одновременные X(t)->Y(t) рёбра.

    Алгоритм:
      1. PC-шаг: условные тесты независимости при растущем множестве соседей.
      2. MCI-шаг: точный тест i(t-tau) _|_ Y(t) | parents(Y), parents(i(t-tau)).
      3. Ориентация одновременных рёбер через v-структуры.

    Тест независимости:
      - ParCorr (по умолчанию): частичная корреляция, предполагает гауссовость.
      - RobustParCorr (use_robust=True): устойчив к тяжёлым хвостам,
        рекомендован для финансовых рядов с выбросами.

    Извлечение backdoor-множества:
      graph[i, j, tau] == "-->" означает X_i(t-tau) -> X_j(t)
      Ищем всех i с ребром i -> target на любом лаге tau in [1, tau_max].
      Сортируем по значимости (min p-value). Исключаем медиаторы.

    Параметры
    ----------
    tau_max    : максимальный лаг (рекомендуется 3-6 для дневных, 3-4 для месячных)
    pc_alpha   : уровень значимости
    use_robust : использовать RobustParCorr вместо ParCorr

    Fallback: при ImportError или ошибке -> expert backdoor set.
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.independence_tests.robust_parcorr import RobustParCorr
    except ImportError:
        log.warning("  tigramite не установлен: pip install tigramite")
        return select_causal_expert(dag), {"method": "expert_fallback_no_tigramite"}

    # ── Подготовка переменных ─────────────────────────────────────────────────
    key_vars = list(dict.fromkeys(
        [target]
        + [c for c in dag.causes[:6]      if c in df.columns]
        + [c for c in dag.confounders[:5] if c in df.columns]
        + [c for c in dag.instruments[:2] if c in df.columns]
    ))

    if len(key_vars) < 3:
        log.warning(f"  [{target}] PCMCI: мало переменных — используем expert DAG")
        return select_causal_expert(dag), {"method": "expert_fallback_too_few_vars"}

    sub = df[key_vars].dropna()

    # Минимум наблюдений: T > 4 * N * tau_max
    min_obs = max(50, 4 * len(key_vars) * tau_max)
    if len(sub) < min_obs:
        log.warning(f"  [{target}] PCMCI: n={len(sub)} < min={min_obs} "
                    f"— используем expert DAG")
        return select_causal_expert(dag), {
            "method": "expert_fallback_insufficient_n",
            "n_obs": len(sub), "min_obs": min_obs,
        }

    log.info(f"  [{target}] PCMCI+: {len(key_vars)} переменных, "
             f"n={len(sub)}, tau_max={tau_max}, alpha={pc_alpha}")

    # ── Нормализация ──────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    data_arr = scaler.fit_transform(sub.values)  # shape: (T, N)

    try:
        test_name = "RobustParCorr" if use_robust else "ParCorr"
        if use_robust:
            from tigramite.independence_tests.robust_parcorr import RobustParCorr
            cond_test = RobustParCorr(significance="analytic")
        else:
            from tigramite.independence_tests.parcorr import ParCorr
            cond_test = ParCorr(significance="analytic")

        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI

        dataframe = pp.DataFrame(data_arr, var_names=key_vars)
        pcmci_obj = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_test,
            verbosity=0,
        )

        # ── Запуск PCMCI+ ─────────────────────────────────────────────────────
        # tau_min=1: только лаговые рёбра X(t-tau)->Y(t), tau>=1
        # Для финансовых рядов предпочтительнее: строже, меньше spurious edges
        results = pcmci_obj.run_pcmciplus(
            tau_min=1,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
        )

        graph      = results["graph"]       # shape: (N, N, tau_max+1)
        p_matrix   = results["p_matrix"]    # shape: (N, N, tau_max+1)
        val_matrix = results["val_matrix"]  # shape: (N, N, tau_max+1)

        target_idx = key_vars.index(target)

        # ── Извлечение родителей целевого узла ───────────────────────────────
        # graph[i, target_idx, tau] == "-->" : X_i(t-tau) -> Y(t)
        parents_info: Dict[str, Dict] = {}
        for i, var in enumerate(key_vars):
            if i == target_idx:
                continue
            for tau in range(1, tau_max + 1):
                if graph[i, target_idx, tau] == "-->":
                    p_val = float(p_matrix[i, target_idx, tau])
                    val   = float(val_matrix[i, target_idx, tau])
                    # Запоминаем лаг с наименьшим p-value
                    if var not in parents_info or p_val < parents_info[var]["p_val"]:
                        parents_info[var] = {
                            "tau": tau, "p_val": p_val, "val": val
                        }

        # Сортируем по значимости
        parents_sorted = sorted(
            parents_info.items(), key=lambda kv: kv[1]["p_val"]
        )
        parent_vars = [var for var, _ in parents_sorted]

        # Исключаем медиаторы (over-control bias)
        parent_vars = [v for v in parent_vars if v not in dag.mediators]

        log.info(
            f"  [{target}] PCMCI+ ({test_name}) родители: "
            + (", ".join(
                f"{v}(tau={d['tau']},p={d['p_val']:.3f})"
                for v, d in parents_sorted[:6]
            ) or "нет")
        )

        # ── Fallback: если ни одного родителя не найдено ─────────────────────
        if not parent_vars:
            log.warning(f"  [{target}] PCMCI+ не нашёл родителей — "
                        f"ищем по p_matrix (порог=0.10)")
            # Смягчённый критерий: берём переменные с min_p < 0.10
            soft_parents = []
            for i, var in enumerate(key_vars):
                if i == target_idx or var in dag.mediators:
                    continue
                min_p = min(
                    float(p_matrix[i, target_idx, tau])
                    for tau in range(1, tau_max + 1)
                )
                if min_p < 0.10:
                    soft_parents.append((var, min_p))
            soft_parents.sort(key=lambda x: x[1])
            parent_vars = [v for v, _ in soft_parents]
            if parent_vars:
                log.info(f"  [{target}] PCMCI+ (soft) родители: {parent_vars}")

        # Последний резерв: expert DAG
        if not parent_vars:
            log.warning(f"  [{target}] PCMCI+ -> expert DAG (нет родителей)")
            expert = select_causal_expert(dag)
            return expert, {
                "method":  "expert_fallback_no_parents",
                "test":    test_name,
                "tau_max": tau_max,
            }

        meta = {
            "method":   f"PCMCI+_{test_name}",
            "tau_max":  tau_max,
            "pc_alpha": pc_alpha,
            "n_vars":   len(key_vars),
            "n_obs":    len(sub),
            "parents":  {
                v: {"tau": d["tau"], "p_val": d["p_val"], "val": d["val"]}
                for v, d in parents_info.items() if v in parent_vars
            },
        }
        return parent_vars, meta

    except Exception as e:
        log.error(f"  [{target}] PCMCI+ ошибка: {e} — используем expert DAG")
        return select_causal_expert(dag), {
            "method": "expert_fallback_exception",
            "error":  str(e),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ ОТБОРА
# ══════════════════════════════════════════════════════════════════════════════

def run_variable_selection(
    use_pcmci: bool  = True,
    tau_max:   int   = 4,
    pc_alpha:           float = 0.05,
    use_robust_parcorr: bool  = False,
) -> Dict[str, Dict]:
    """
    Запускает все три метода отбора для каждого из 11 активов.

    Параметры
    ----------
    use_pcmci           : запускать PCMCI+ (требует tigramite)
    tau_max   : максимальный лаг PCMCI+ (Метод C)
    pc_alpha            : уровень значимости PCMCI+
    use_robust_parcorr  : использовать RobustParCorr в PCMCI+

    Структура возвращаемого словаря и JSON:
    {
      "WTI_oil": {
        "heuristic":      [...],  # Метод A: LassoCV + PCA
        "causal_expert":  [...],  # Метод B: expert DAG backdoor
        "causal_pc":      [...],  # Метод C: PCMCI+
        "causal_union":   [...],  # B union C (приоритет эксперта)
        "causes":         [...],
        "instruments":    [...],
        "freq":           "D",
        "file":           "A_WTI_oil.csv",
        "selection_meta": {
          "heuristic": {lasso_alpha, pca_n_components, ...},
          "pcmci":     {method, parents, tau_max, ...}
        }
      }, ...
    }
    """
    all_results: Dict[str, Dict] = {}

    for target, dag in ALL_DAGS.items():
        log.info(f"\n{'─'*55}")
        log.info(f"  {dag.name} ({target})  [freq={dag.freq}]")
        log.info(f"{'─'*55}")

        df_raw = load_dataset(dag)
        if df_raw.empty or target not in df_raw.columns:
            log.warning(f"  Пропускаем {target}: нет данных")
            continue

        df_stat = prepare_stationary(df_raw, target)

        # ── Метод A: LassoCV + PCA ────────────────────────────────────────────
        controls_heuristic, meta_heuristic = select_heuristic(df_stat, target)

        # ── Метод B: Expert DAG ───────────────────────────────────────────────
        controls_expert = select_causal_expert(dag)

        # ── Метод C: PCMCI+ ───────────────────────────────────────────────────
        if use_pcmci:
            # Адаптируем tau_max к частоте:
            # дневные -> tau_max, месячные -> min(tau_max, 3)
            effective_tau = tau_max if dag.freq != "ME" else min(tau_max, 3)
            controls_pcmci, meta_pcmci = select_causal_pcmci(
                df_stat, target, dag,
                tau_max=effective_tau,
                pc_alpha=pc_alpha,
                use_robust=use_robust_parcorr,
            )
        else:
            controls_pcmci = controls_expert
            meta_pcmci     = {"method": "expert_fallback_no_pcmci"}

        # ── Union: Expert union PCMCI (приоритет эксперта) ───────────────────
        controls_union = list(dict.fromkeys(
            controls_expert
            + [v for v in controls_pcmci if v not in controls_expert]
        ))

        # ── Сериализация ──────────────────────────────────────────────────────
        def _jsonify(obj):
            if isinstance(obj, dict):
                return {k: _jsonify(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_jsonify(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return obj

        all_results[target] = {
            "heuristic":      controls_heuristic,
            "causal_expert":  controls_expert,
            "causal_pc":      controls_pcmci,
            "causal_union":   controls_union,
            "causes":         dag.causes,
            "instruments":    dag.instruments,
            "freq":           dag.freq,
            "file":           dag.file,
            "selection_meta": _jsonify({
                "heuristic": meta_heuristic,
                "pcmci":     meta_pcmci,
            }),
        }

        log.info(f"  -> Heuristic (LassoCV+PCA) : {controls_heuristic}")
        log.info(f"  -> Expert DAG              : {controls_expert}")
        log.info(f"  -> PCMCI+                  : {controls_pcmci}")
        log.info(f"  -> Union (expert + pcmci)  : {controls_union}")

    # ── Сохраняем JSON ────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "step2_variable_selection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    log.info(f"\n Результаты сохранены -> {out_path}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 2: Variable selection (Heuristic + Expert + PCMCI+)"
    )
    parser.add_argument("--no-pcmci",          action="store_true",
                        help="Отключить PCMCI+, использовать только expert DAG")
    parser.add_argument("--tau-max",            type=int,   default=4,
                        help="Максимальный лаг PCMCI+ (по умолч. 4)")
    parser.add_argument("--alpha",              type=float, default=0.05,
                        help="Уровень значимости PCMCI+ (по умолч. 0.05)")
    parser.add_argument("--robust",  action="store_true",
                        help="RobustParCorr вместо ParCorr в PCMCI+")
    args = parser.parse_args()

    results = run_variable_selection(
        use_pcmci          = not args.no_pcmci,
        tau_max            = args.tau_max,
        pc_alpha           = args.alpha,
        use_robust_parcorr = args.robust,
    )

    print("\n" + "=" * 62)
    print("  ИТОГ: контрольные наборы переменных")
    print("=" * 62)
    for target, info in results.items():
        meta_h = info.get("selection_meta", {}).get("heuristic", {})
        meta_p = info.get("selection_meta", {}).get("pcmci", {})
        print(f"\n  {target}  [{info['freq']}]")
        print(f"    Heuristic  (n={meta_h.get('n_final','?')}, "
              f"alpha_lasso={meta_h.get('lasso_alpha','?')}): "
              f"{info['heuristic']}")
        print(f"    Expert DAG : {info['causal_expert']}")
        print(f"    PCMCI+     ({meta_p.get('method','?')}): {info['causal_pc']}")

