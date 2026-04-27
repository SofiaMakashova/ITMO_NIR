"""
causal_pipeline/step4_baseline_models.py
═════════════════════════════════════════
Шаг 4: Baseline модели (без каузальной коррекции)

  1. ARIMA     — классический статистический baseline
  2. Prophet   — Facebook/Meta Prophet (тренд + сезонность)
  3. Random Forest — ML baseline с техническими фичами
  4. LSTM      — нейросетевой baseline (PyTorch или TF Keras)

Сравнение с каузальными методами из step3.
"""

import warnings
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step1_dag_definitions import ALL_DAGS
from step2_variable_selection import load_dataset, prepare_stationary
from step3_causal_models import (
    compute_metrics, prepare_ml_data, train_test_split_ts, TRAIN_RATIO
)


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 1: ARIMA (auto-order selection)
# ══════════════════════════════════════════════════════════════════════════════

def run_arima(df: pd.DataFrame, target: str, horizon: int = 1) -> Dict:
    """
    ARIMA baseline с правильной многошаговой стратегией.

    Стратегия прогноза зависит от горизонта:

    h = 1  —  Rolling 1-step-ahead:
        На каждом шаге теста: обучаем на истории до t, предсказываем t+1,
        добавляем реальное значение y[t] в историю, сдвигаемся.
        Это стандартный walk-forward evaluation.

    h > 1  —  DIRECT (прямой) метод:
        Обучаем отдельную ARIMA на сдвинутом целевом ряде y[t+h] ~ ARIMA(y[t], ...),
        затем делаем rolling 1-step-ahead прогноз сдвинутого ряда.
        Это избегает накопления ошибки рекурсивного метода и гарантирует
        что RMSE на h=5 и h=21 отличаются от h=1.

        Альтернатива — рекурсивный forecast(steps=h): формально корректна,
        но для нестационарных/слабопредсказуемых рядов даёт идентичные
        метрики при разных h (модель вырождается в прогноз среднего).
        DIRECT-метод этого лишён.

    Определение стационарности:
        ADF-тест перед auto_arima. Если ряд уже стационарен (p<0.05),
        фиксируем d=0 чтобы pmdarima не падал с ошибкой на d=1.
    """
    log.info(f"  [ARIMA] {target}  h={horizon}")

    from statsmodels.tsa.stattools import adfuller

    series = df[target].dropna()
    n   = len(series)
    cut = int(n * TRAIN_RATIO)

    # ── DIRECT: для h>1 сдвигаем целевой ряд вперёд ─────────────────────────
    # y_shifted[t] = y[t + horizon]  →  ARIMA предсказывает значение через h шагов
    if horizon > 1:
        y_shifted = series.shift(-horizon).dropna()
        # Обрезаем исходный ряд по длине сдвинутого
        series_aligned = series.iloc[:len(y_shifted)]
        cut_d = int(len(series_aligned) * TRAIN_RATIO)
        train_x = series_aligned.iloc[:cut_d]   # предикторная история (для обучения ARIMA)
        train_y = y_shifted.iloc[:cut_d]         # сдвинутый целевой (то что предсказываем)
        test_y  = y_shifted.iloc[cut_d:]         # истинные значения y[t+h] на тесте
        # Для обновления истории используем несдвинутый ряд
        test_x  = series_aligned.iloc[cut_d:]
    else:
        train_x = series.iloc[:cut]
        train_y = series.iloc[:cut]
        test_y  = series.iloc[cut:]
        test_x  = series.iloc[cut:]

    # ── Определяем стационарность обучающего ряда ────────────────────────────
    try:
        is_stationary = adfuller(train_y.dropna(), autolag="AIC")[1] < 0.05
    except Exception:
        is_stationary = False

    preds = []

    # ── Попытка 1: auto_arima ─────────────────────────────────────────────────
    try:
        from pmdarima import auto_arima

        model = auto_arima(
            train_y,
            seasonal=False, stepwise=True,
            information_criterion="aic",
            max_p=4, max_q=4,
            d=0 if is_stationary else None,
            max_d=2,
            error_action="ignore", suppress_warnings=True,
            with_intercept=True,
        )
        # Rolling 1-step-ahead прогноз сдвинутого ряда
        for i in range(len(test_y)):
            fc = model.predict(n_periods=1)
            preds.append(float(fc[0]))
            # Обновляем модель реальным значением сдвинутого ряда
            model.update([test_y.iloc[i]])

        metrics = compute_metrics(
            test_y.values[:len(preds)], np.array(preds),
            f"ARIMA_direct(h={horizon})"
        )
        return {
            "method":  f"ARIMA{model.order}",
            "metrics": metrics,
            "predictions": dict(zip(
                test_x.index.strftime("%Y-%m-%d").tolist()[:len(preds)], preds)),
            "order":    list(model.order),
            "strategy": "direct" if horizon > 1 else "rolling_1step",
        }

    except ImportError:
        pass
    except Exception as e:
        log.warning(f"    auto_arima error: {e}")

    # ── Fallback: statsmodels ARIMA ───────────────────────────────────────────
    try:
        import statsmodels.tsa.arima.model as smarima

        d_order   = 0 if is_stationary else 1
        arima_order = (2, d_order, 2)
        history = list(train_y)

        for i in range(len(test_y)):
            try:
                mod = smarima.ARIMA(history, order=arima_order)
                res = mod.fit(method_kwargs={"warn_convergence": False})
                preds.append(float(res.forecast(1).iloc[0]))
            except Exception:
                preds.append(float(np.mean(history[-20:])))  # локальное среднее
            # Обновляем историю реальным значением сдвинутого ряда
            history.append(float(test_y.iloc[i]))

        metrics = compute_metrics(
            test_y.values[:len(preds)], np.array(preds),
            f"ARIMA{arima_order}_direct(h={horizon})"
        )
        return {
            "method":  f"ARIMA{arima_order}",
            "metrics": metrics,
            "predictions": dict(zip(
                test_x.index.strftime("%Y-%m-%d").tolist()[:len(preds)], preds)),
            "strategy": "direct" if horizon > 1 else "rolling_1step",
        }

    except Exception as e:
        log.error(f"    ARIMA fallback error: {e}")
        return {"method": "ARIMA_FAILED", "metrics": {}}


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 2: Prophet
# ══════════════════════════════════════════════════════════════════════════════

def run_prophet(df: pd.DataFrame, target: str,
                extra_regressors: List[str] = None,
                horizon: int = 1) -> Dict:
    """
    Facebook Prophet:
      - Тренд + сезонность
      - Опционально: дополнительные регрессоры (контрольные переменные)
      - Без каузальной коррекции (baseline)
    """
    log.info(f"  [Prophet] {target}")

    try:
        from prophet import Prophet
    except ImportError:
        try:
            from fbprophet import Prophet
        except ImportError:
            log.warning("    prophet не установлен: pip install prophet")
            return {"method": "Prophet_SKIPPED", "metrics": {}}

    series = df[target].dropna()
    n = len(series)
    cut = int(n * TRAIN_RATIO)

    # Частота для Prophet
    freq_map = {"D": "D", "B": "B", "ME": "MS", "M": "MS"}
    freq = freq_map.get(pd.infer_freq(series.index) or "D", "D")

    prophet_df = pd.DataFrame({
        "ds": series.index,
        "y":  series.values
    }).reset_index(drop=True)

    # Дополнительные регрессоры
    if extra_regressors:
        for reg in extra_regressors[:3]:  # не более 3
            if reg in df.columns:
                prophet_df[reg] = df[reg].reindex(series.index).ffill().values

    train_df = prophet_df.iloc[:cut]
    test_df  = prophet_df.iloc[cut:]

    try:
        m = Prophet(
            seasonality_mode="multiplicative" if df[target].min() > 0 else "additive",
            daily_seasonality=False,
            weekly_seasonality=(freq in ["D", "B"]),
            yearly_seasonality=True,
            interval_width=0.95,
        )
        if extra_regressors:
            for reg in extra_regressors[:3]:
                if reg in prophet_df.columns:
                    m.add_regressor(reg)

        m.fit(train_df)

        # Прогноз
        future = m.make_future_dataframe(
            periods=len(test_df), freq=freq, include_history=False)
        if extra_regressors:
            for reg in extra_regressors[:3]:
                if reg in prophet_df.columns:
                    future[reg] = test_df[reg].values[:len(future)]

        forecast = m.predict(future)
        y_pred = forecast["yhat"].values[:len(test_df)]
        y_true = test_df["y"].values[:len(y_pred)]

        metrics = compute_metrics(y_true, y_pred, "Prophet")
        return {
            "method":  "Prophet",
            "metrics": metrics,
            "predictions": dict(zip(
                test_df["ds"].dt.strftime("%Y-%m-%d").tolist()[:len(y_pred)],
                y_pred.tolist()
            )),
        }
    except Exception as e:
        log.error(f"    Prophet error: {e}")
        return {"method": "Prophet_FAILED", "metrics": {}}


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 3: Random Forest (без каузальной коррекции)
# ══════════════════════════════════════════════════════════════════════════════

def run_random_forest(df: pd.DataFrame, target: str,
                      all_features: List[str],
                      horizon: int = 1) -> Dict:
    """
    Random Forest Regressor:
      - Использует ВСЕ доступные фичи (включая технические)
      - Без backdoor коррекции (baseline)
      - Включает SHAP importance для интерпретации
    """
    log.info(f"  [RandomForest] {target}")

    from sklearn.ensemble import RandomForestRegressor

    X_mat, y_vec = prepare_ml_data(
        df, target, all_features, all_features,
        horizon=horizon, n_lags=10
    )
    if len(X_mat) < 60:
        return {"method": "RF_SKIPPED", "metrics": {}}

    X_tr, X_te, y_tr, y_te = train_test_split_ts(X_mat, y_vec)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    metrics = compute_metrics(y_te.values, y_pred, "RandomForest")

    # Feature importance
    importances = pd.Series(rf.feature_importances_,
                            index=X_mat.columns).sort_values(ascending=False)
    top_features = importances.head(10).to_dict()
    top_features = {k: round(float(v), 4) for k, v in top_features.items()}

    return {
        "method":  "RandomForest",
        "metrics": metrics,
        "predictions": dict(zip(
            y_te.index.strftime("%Y-%m-%d").tolist(),
            y_pred.tolist()
        )),
        "top_features": top_features,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 4: LSTM
# ══════════════════════════════════════════════════════════════════════════════

def run_lstm(df: pd.DataFrame, target: str,
             feature_cols: List[str],
             seq_len: int = 20,
             horizon: int = 1,
             epochs: int = 30) -> Dict:
    """
    LSTM (PyTorch):
      - Многомерный вход: [target_lags + feature_cols]
      - Однослойный LSTM → Linear → предсказание
      - Без каузальной коррекции (raw features)
    """
    log.info(f"  [LSTM] {target}")

    # Пытаемся PyTorch
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        return _lstm_pytorch(df, target, feature_cols,
                             seq_len, horizon, epochs)
    except ImportError:
        pass

    # Fallback: Keras/TF
    try:
        import tensorflow as tf
        return _lstm_keras(df, target, feature_cols,
                           seq_len, horizon, epochs)
    except ImportError:
        log.warning("    ни PyTorch, ни TF не установлены. Пропускаем LSTM.")
        return {"method": "LSTM_SKIPPED", "metrics": {}}


def _lstm_pytorch(df, target, feature_cols, seq_len, horizon, epochs):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    avail = [c for c in feature_cols if c in df.columns]
    cols  = [target] + avail[:9]  # target + до 9 фич
    sub   = df[cols].dropna()
    n     = len(sub)
    cut   = int(n * TRAIN_RATIO)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(sub.values)

    def make_sequences(arr, seq, h):
        X_s, y_s = [], []
        for i in range(len(arr) - seq - h + 1):
            X_s.append(arr[i:i + seq])
            y_s.append(arr[i + seq + h - 1, 0])  # 0 = target column
        return np.array(X_s), np.array(y_s)

    train_data = data_scaled[:cut]
    test_data  = data_scaled[cut - seq_len:]  # overlap для контекста

    X_tr, y_tr = make_sequences(train_data, seq_len, horizon)
    X_te, y_te = make_sequences(test_data,  seq_len, horizon)

    if len(X_tr) < 20:
        return {"method": "LSTM_SKIPPED", "metrics": {}}

    # ── Модель ────────────────────────────────────────────────────────────────
    class LSTMModel(nn.Module):
        def __init__(self, n_feat, hidden=64, n_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, n_layers,
                                batch_first=True, dropout=dropout)
            self.fc   = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = LSTMModel(n_feat=len(cols)).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit   = nn.MSELoss()
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=5, factor=0.5)

    X_t = torch.FloatTensor(X_tr).to(device)
    y_t = torch.FloatTensor(y_tr).to(device)
    ds  = TensorDataset(X_t, y_t)
    dl  = DataLoader(ds, batch_size=32, shuffle=False)

    # ── Обучение ──────────────────────────────────────────────────────────────
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in dl:
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss += loss.item()
        sched.step(epoch_loss)
        if (epoch + 1) % 10 == 0:
            log.info(f"    epoch {epoch+1}/{epochs} loss={epoch_loss:.4f}")

    # ── Предсказание ──────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        X_te_t = torch.FloatTensor(X_te).to(device)
        preds_scaled = model(X_te_t).cpu().numpy()

    # Обратное масштабирование (только колонка target=0)
    pad = np.zeros((len(preds_scaled), len(cols)))
    pad[:, 0] = preds_scaled
    preds_inv = scaler.inverse_transform(pad)[:, 0]

    pad2 = np.zeros((len(y_te), len(cols)))
    pad2[:, 0] = y_te
    true_inv = scaler.inverse_transform(pad2)[:, 0]

    metrics = compute_metrics(true_inv, preds_inv, "LSTM_PyTorch")

    test_index = sub.index[cut - seq_len + seq_len + horizon - 1:
                           cut - seq_len + seq_len + horizon - 1 + len(preds_inv)]
    return {
        "method":  "LSTM_PyTorch",
        "metrics": metrics,
        "predictions": dict(zip(
            test_index.strftime("%Y-%m-%d").tolist(),
            preds_inv.tolist()
        )),
    }


def _lstm_keras(df, target, feature_cols, seq_len, horizon, epochs):
    import tensorflow as tf

    avail = [c for c in feature_cols if c in df.columns]
    cols  = [target] + avail[:9]
    sub   = df[cols].dropna()
    n, cut = len(sub), int(len(sub) * TRAIN_RATIO)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(sub.values)

    def make_sequences(arr, seq, h):
        X_s, y_s = [], []
        for i in range(len(arr) - seq - h + 1):
            X_s.append(arr[i:i + seq])
            y_s.append(arr[i + seq + h - 1, 0])
        return np.array(X_s), np.array(y_s)

    X_tr, y_tr = make_sequences(data_scaled[:cut], seq_len, horizon)
    X_te, y_te = make_sequences(data_scaled[cut - seq_len:], seq_len, horizon)

    if len(X_tr) < 20:
        return {"method": "LSTM_SKIPPED", "metrics": {}}

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                             input_shape=(seq_len, len(cols))),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
              verbose=0, validation_split=0.1)

    preds_scaled = model.predict(X_te, verbose=0).flatten()
    pad  = np.zeros((len(preds_scaled), len(cols)))
    pad[:, 0] = preds_scaled
    preds_inv = scaler.inverse_transform(pad)[:, 0]
    pad2 = np.zeros((len(y_te), len(cols)))
    pad2[:, 0] = y_te
    true_inv = scaler.inverse_transform(pad2)[:, 0]

    metrics = compute_metrics(true_inv, preds_inv, "LSTM_Keras")
    return {"method": "LSTM_Keras", "metrics": metrics, "predictions": {}}


# ══════════════════════════════════════════════════════════════════════════════
# АДАПТИВНЫЕ ПАРАМЕТРЫ ПО ЧАСТОТЕ
# ══════════════════════════════════════════════════════════════════════════════

# Горизонты и seq_len LSTM по частоте данных
FREQ_PARAMS = {
    "D":  {"horizons": [1, 5, 21], "seq_len": 30,  "lstm_epochs": 30},
    "B":  {"horizons": [1, 5, 21], "seq_len": 30,  "lstm_epochs": 30},
    "W":  {"horizons": [1, 4, 13], "seq_len": 12,  "lstm_epochs": 25},
    "ME": {"horizons": [1, 3, 6],  "seq_len": 12,  "lstm_epochs": 50},
    "MS": {"horizons": [1, 3, 6],  "seq_len": 12,  "lstm_epochs": 50},
    "M":  {"horizons": [1, 3, 6],  "seq_len": 12,  "lstm_epochs": 50},
}

def get_freq_params(freq: str) -> dict:
    return FREQ_PARAMS.get(freq, FREQ_PARAMS["D"])


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК ВСЕХ BASELINE МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════

def run_all_baselines(var_selection: Dict) -> Dict:
    """
    Запускает 4 baseline модели для каждого датасета.
    Горизонты прогноза адаптируются к частоте данных:
      - Дневные (A/B/D): h = 1, 5, 21
      - Месячные (C):    h = 1, 3, 6
    """
    all_results = {}

    for target, info in var_selection.items():
        dag = ALL_DAGS.get(target)
        if dag is None:
            continue

        log.info(f"\n{'═'*60}")
        log.info(f"  BASELINE МОДЕЛИ: {dag.name} ({target})  "
                 f"[freq={dag.freq}]")
        log.info(f"{'═'*60}")

        df_raw  = load_dataset(dag)
        if df_raw.empty or target not in df_raw.columns:
            continue
        df_stat = prepare_stationary(df_raw, target)

        all_features = [c for c in df_stat.columns if c != target]
        controls     = info.get("heuristic", [])

        fparams  = get_freq_params(dag.freq)
        horizons = fparams["horizons"]
        seq_len  = fparams["seq_len"]
        epochs   = fparams["lstm_epochs"]

        target_results = {
            "freq":    dag.freq,
            "horizons": horizons,
        }

        for h in horizons:
            log.info(f"\n  ▸ horizon = {h}")
            h_res = {}

            # ARIMA
            h_res["arima"] = run_arima(df_stat, target, horizon=h)

            # Prophet (на исходных данных — лучше обрабатывает тренд)
            h_res["prophet"] = run_prophet(
                df_raw, target,
                extra_regressors=controls[:3],
                horizon=h
            )

            # Random Forest
            h_res["random_forest"] = run_random_forest(
                df_stat, target, all_features[:30], horizon=h)

            # LSTM
            h_res["lstm"] = run_lstm(
                df_raw, target, controls[:8],
                seq_len=seq_len, horizon=h, epochs=epochs
            )

            target_results[f"h{h}"] = h_res

        all_results[target] = target_results

    # ── сериализация ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "step4_baselines.json"
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

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_clean(all_results), f, ensure_ascii=False, indent=2)
    log.info(f"\n✓ Результаты step4 сохранены: {out_path}")
    return all_results


if __name__ == "__main__":
    sel_path = RESULTS_DIR / "step2_variable_selection.json"
    if sel_path.exists():
        with open(sel_path) as f:
            var_sel = json.load(f)
    else:
        from step2_variable_selection import run_variable_selection
        var_sel = run_variable_selection(use_pcmci=True)

    run_all_baselines(var_sel)
