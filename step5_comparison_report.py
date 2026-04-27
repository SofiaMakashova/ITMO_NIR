"""
causal_pipeline/step5_comparison_report.py
═══════════════════════════════════════════
Шаг 5: Сравнение каузальных и baseline методов

Новая структура данных из step3/step4:
  step3: {target → {controls_heuristic/expert/pc → {hN → {method → result}}}}
  step4: {target → {hN → {method → result}}}

Выходы:
  ① results/full_comparison.csv      — все метрики по всем осям
  ② results/summary_h1_RMSE.csv      — сводная таблица на горизонте h=1
  ③ results/dag_comparison.csv       — expert vs pc: где DAG-метод лучше?
  ④ figures/heatmap_RMSE_hN.png      — тепловые карты по горизонтам
  ⑤ figures/causal_vs_baseline_*.png — grouped bar charts
  ⑥ figures/dag_divergence_RMSE.png  — расхождение expert / PC DAG
  ⑦ figures/horizon_progression.png  — деградация метрики по горизонтам
  ⑧ Консольный итоговый отчёт
"""

import json
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── константы ─────────────────────────────────────────────────────────────────
CAUSAL_METHODS   = ["causal_impact", "dml", "var_granger", "iv_2sls"]
BASELINE_METHODS = ["arima", "prophet", "random_forest", "lstm"]
CONTROL_SETS     = ["controls_heuristic", "controls_causal_expert",
                    "controls_causal_pc"]

METHOD_LABELS = {
    "causal_impact":  "CausalImpact",
    "dml":            "DML",
    "var_granger":    "VAR+Granger",
    "iv_2sls":        "IV-2SLS",
    "arima":          "ARIMA",
    "prophet":        "Prophet",
    "random_forest":  "RandomForest",
    "lstm":           "LSTM",
}
CTRL_LABELS = {
    "controls_heuristic":     "Heuristic-Z",
    "controls_causal_expert": "Expert-DAG",
    "controls_causal_pc":     "PC-DAG",
}
ASSET_LABELS = {
    "WTI_oil": "WTI Oil",  "NatGas":  "Nat.Gas",
    "Gold":    "Gold",     "EURUSD":  "EUR/USD",
    "GBPUSD":  "GBP/USD",  "USDJPY":  "USD/JPY",
    "CPI":     "CPI",      "IndProd": "IndProd",
    "UMCSENT": "UMCSENT",  "BTC":     "Bitcoin",
    "ETH":     "Ethereum",
}
ASSET_GROUP = {
    "WTI_oil": "A", "NatGas": "A", "Gold": "A",
    "EURUSD":  "B", "GBPUSD": "B", "USDJPY": "B",
    "CPI":     "C", "IndProd": "C", "UMCSENT": "C",
    "BTC":     "D", "ETH": "D",
}
GROUP_COLORS = {
    "A": "#FF6B6B",
    "B": "#4ECDC4",
    "C": "#45B7D1",
    "D": "#FFA07A",
}
COLORS_CAUSAL   = ["#2196F3", "#00BCD4", "#4CAF50", "#8BC34A"]
COLORS_BASELINE = ["#FF9800", "#F44336", "#9C27B0", "#795548"]

_DARK_BG  = "#FFFFFF"
_CARD_BG  = "#F5F5F5"
_GRID_CLR = "#E0E0E0"


# ══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА И ПАРСИНГ
# ══════════════════════════════════════════════════════════════════════════════

def load_results() -> Tuple[Dict, Dict]:
    c_path = RESULTS_DIR / "step3_causal_models.json"
    b_path = RESULTS_DIR / "step4_baselines.json"
    causal   = json.loads(c_path.read_text()) if c_path.exists() else {}
    baseline = json.loads(b_path.read_text()) if b_path.exists() else {}
    return causal, baseline


def build_full_df(causal: Dict, baseline: Dict) -> pd.DataFrame:
    """
    Flatten всех результатов в единый DataFrame:
    (Asset, Group, Freq, Horizon, ControlSet, Method, Type, RMSE, MAE, MAPE, R2)
    """
    rows = []

    # ── Step 3: каузальные ────────────────────────────────────────────────────
    for target, tdata in causal.items():
        if not isinstance(tdata, dict):
            continue
        freq     = tdata.get("freq", "D")
        horizons = tdata.get("horizons", [1])

        for ctrl_key in CONTROL_SETS:
            ctrl_data = tdata.get(ctrl_key, {})
            if not isinstance(ctrl_data, dict):
                continue
            for h in horizons:
                hkey   = f"h{h}"
                h_data = ctrl_data.get(hkey, {})
                if not isinstance(h_data, dict):
                    continue
                for method in CAUSAL_METHODS:
                    m_data  = h_data.get(method, {})
                    metrics = m_data.get("metrics", {}) if isinstance(m_data, dict) else {}
                    rows.append({
                        "Asset":      ASSET_LABELS.get(target, target),
                        "Target":     target,
                        "Group":      ASSET_GROUP.get(target, "?"),
                        "Freq":       freq,
                        "Horizon":    h,
                        "ControlSet": CTRL_LABELS.get(ctrl_key, ctrl_key),
                        "Method":     METHOD_LABELS.get(method, method),
                        "Type":       "Causal",
                        "RMSE":       metrics.get("RMSE"),
                        "MAE":        metrics.get("MAE"),
                        "MAPE":       metrics.get("MAPE"),
                        "R2":         metrics.get("R2"),
                    })

    # ── Step 4: baseline ──────────────────────────────────────────────────────
    for target, tdata in baseline.items():
        if not isinstance(tdata, dict):
            continue
        freq     = tdata.get("freq", "D")
        horizons = tdata.get("horizons", [1])

        for h in horizons:
            hkey   = f"h{h}"
            h_data = tdata.get(hkey, {})
            if not isinstance(h_data, dict):
                continue
            for method in BASELINE_METHODS:
                m_data  = h_data.get(method, {})
                metrics = m_data.get("metrics", {}) if isinstance(m_data, dict) else {}
                rows.append({
                    "Asset":      ASSET_LABELS.get(target, target),
                    "Target":     target,
                    "Group":      ASSET_GROUP.get(target, "?"),
                    "Freq":       freq,
                    "Horizon":    h,
                    "ControlSet": "Baseline (no Z)",
                    "Method":     METHOD_LABELS.get(method, method),
                    "Type":       "Baseline",
                    "RMSE":       metrics.get("RMSE"),
                    "MAE":        metrics.get("MAE"),
                    "MAPE":       metrics.get("MAPE"),
                    "R2":         metrics.get("R2"),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["RMSE", "MAE", "MAPE", "R2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_summary_h1(full_df: pd.DataFrame, metric: str = "RMSE") -> pd.DataFrame:
    """Сводная таблица на горизонте h=1."""
    h1 = full_df[full_df["Horizon"] == 1].copy()

    causal_h1  = h1[h1["Type"] == "Causal"]
    baseline_h1 = h1[h1["Type"] == "Baseline"]

    pivot_c = causal_h1.pivot_table(
        index="Asset", columns=["ControlSet", "Method"],
        values=metric, aggfunc="mean"
    )
    pivot_b = baseline_h1.pivot_table(
        index="Asset", columns="Method", values=metric, aggfunc="mean"
    )
    if not pivot_c.empty:
        pivot_c.columns = [f"{c} | {m}" for c, m in pivot_c.columns]
    return pivot_c.join(pivot_b, how="outer") if not pivot_c.empty else pivot_b


def build_dag_comparison(full_df: pd.DataFrame, metric: str = "RMSE") -> pd.DataFrame:
    """Сравнение Expert-DAG vs PC-DAG."""
    expert = (full_df[full_df["ControlSet"] == "Expert-DAG"]
              [["Target", "Horizon", "Method", metric]]
              .rename(columns={metric: "expert"}))
    pc = (full_df[full_df["ControlSet"] == "PC-DAG"]
          [["Target", "Horizon", "Method", metric]]
          .rename(columns={metric: "pc"}))

    merged = expert.merge(pc, on=["Target", "Horizon", "Method"], how="inner")
    merged["diff_pc_minus_expert"] = merged["pc"] - merged["expert"]
    merged["pc_wins"] = merged["diff_pc_minus_expert"] < 0
    merged["Asset"]   = merged["Target"].map(ASSET_LABELS)
    return merged.sort_values("diff_pc_minus_expert")


# ══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ ВИЗУАЛИЗАЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def _dark_fig(w, h):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(_DARK_BG)
    return fig


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(_CARD_BG)
    ax.set_title(title, color="black", fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color="#AAAAAA", fontsize=9)
    ax.set_ylabel(ylabel, color="#AAAAAA", fontsize=9)
    ax.tick_params(colors="black", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID_CLR)
    ax.grid(color=_GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.7)


# ══════════════════════════════════════════════════════════════════════════════
# ГРАФИКИ
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap_by_horizon(full_df: pd.DataFrame, metric: str = "RMSE",
                            horizon: int = 1):
    """Heatmap: активы × методы, цвет = нормированный RMSE."""
    sub = full_df[full_df["Horizon"] == horizon].copy()
    if sub.empty:
        return

    causal_best = (sub[sub["Type"] == "Causal"]
                   .groupby(["Asset", "Method"])[metric].min().reset_index())
    baseline_sub = sub[sub["Type"] == "Baseline"][["Asset", "Method", metric]]
    combined = pd.concat([causal_best, baseline_sub], ignore_index=True)

    pivot = combined.pivot_table(index="Asset", columns="Method",
                                 values=metric, aggfunc="mean")

    col_order = ([METHOD_LABELS[m] for m in CAUSAL_METHODS
                  if METHOD_LABELS[m] in pivot.columns] +
                 [METHOD_LABELS[m] for m in BASELINE_METHODS
                  if METHOD_LABELS[m] in pivot.columns])
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]

    nr, nc = pivot.shape
    fig = _dark_fig(max(14, nc * 1.7), max(5, nr * 0.7))
    ax  = fig.add_subplot(111)

    pivot_norm = pivot.div(pivot.max(axis=1) + 1e-9, axis=0)
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#27AE60", "#F39C12", "#E74C3C"], N=256)
    im = ax.imshow(pivot_norm.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    for i in range(nr):
        for j in range(nc):
            val = pivot.iloc[i, j]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                best_row = pivot.iloc[i].min()
                fw = "bold" if abs(val - best_row) < 1e-9 else "normal"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color="black", fontweight=fw)

    ax.set_xticks(range(nc))
    ax.set_yticks(range(nr))
    ax.set_xticklabels(col_order, rotation=35, ha="right",
                       fontsize=9, color="black")
    ylabels = ax.set_yticklabels(pivot.index, fontsize=9)
    for lbl in ylabels:
        akey = next((k for k, v in ASSET_LABELS.items()
                     if v == lbl.get_text()), None)
        lbl.set_color(GROUP_COLORS.get(ASSET_GROUP.get(akey, ""), "black"))

    n_c = len([c for c in col_order
               if c in [METHOD_LABELS[m] for m in CAUSAL_METHODS]])
    if 0 < n_c < nc:
        ax.axvline(n_c - 0.5, color="#FFD700", linewidth=2, linestyle="--")
        ax.text(n_c / 2 - 0.5, -0.9, "◀ Causal",
                ha="center", fontsize=8, color="#FFD700",
                transform=ax.get_xaxis_transform())
        ax.text(n_c + (nc - n_c) / 2 - 0.5, -0.9, "Baseline ▶",
                ha="center", fontsize=8, color="#FF8C00",
                transform=ax.get_xaxis_transform())

    ax.set_title(f"Сравнение методов — {metric}, h={horizon}  "
                 f"(зелёный = лучше | нормировка построчная)",
                 color="black", fontsize=12, pad=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(colors="black", labelsize=7)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f"Группа {g}")
                       for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper left",
              facecolor=_CARD_BG, edgecolor=_GRID_CLR,
              labelcolor="black", fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / f"heatmap_{metric}_h{horizon}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    log.info(f"  ✓ {path.name}")
    plt.close()


def plot_horizon_progression(full_df: pd.DataFrame, metric: str = "RMSE"):
    """Линейный график деградации метрики по горизонтам для каждого актива."""
    targets = full_df["Target"].dropna().unique().tolist()
    if not targets:
        return

    n = len(targets)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = _dark_fig(ncols * 6, nrows * 4)
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.5, wspace=0.35)

    all_methods = ([METHOD_LABELS[m] for m in CAUSAL_METHODS] +
                   [METHOD_LABELS[m] for m in BASELINE_METHODS])
    colors = COLORS_CAUSAL + COLORS_BASELINE
    ls_map = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    for idx, tgt in enumerate(targets):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        _style_ax(ax, title=ASSET_LABELS.get(tgt, tgt),
                  xlabel="Горизонт", ylabel=metric)

        sub = full_df[full_df["Target"] == tgt]
        c_best = (sub[sub["Type"] == "Causal"]
                  .groupby(["Horizon", "Method"])[metric].min().reset_index())
        b_sub  = sub[sub["Type"] == "Baseline"][["Horizon", "Method", metric]]
        combined = pd.concat([c_best, b_sub], ignore_index=True)

        for i, (method, color) in enumerate(zip(all_methods, colors)):
            mdata = combined[combined["Method"] == method].sort_values("Horizon")
            if mdata.empty or mdata[metric].isna().all():
                continue
            ax.plot(mdata["Horizon"], mdata[metric],
                    marker="o", markersize=4, linewidth=1.6,
                    color=color, linestyle=ls_map[i % 8],
                    label=method, alpha=0.9)

        ax.legend(fontsize=5.5, facecolor=_CARD_BG, edgecolor=_GRID_CLR,
                  labelcolor="black", ncol=2)

    fig.suptitle(f"Деградация метрики {metric} по горизонтам прогноза",
                 color="black", fontsize=13, y=1.02)
    path = FIGURES_DIR / f"horizon_progression_{metric}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    log.info(f"  ✓ {path.name}")
    plt.close()


def plot_dag_divergence(dag_df: pd.DataFrame, metric: str = "RMSE"):
    """Diverging bar chart: Expert-DAG vs PC-DAG."""
    if dag_df.empty:
        return

    agg = dag_df.groupby("Target")["diff_pc_minus_expert"].mean().sort_values()

    fig = _dark_fig(10, max(5, len(agg) * 0.6))
    ax  = fig.add_subplot(111)
    _style_ax(ax,
              title=f"Expert-DAG vs PC-DAG: Δ{metric} (PC − Expert)\n"
                    f"зелёный (отрицательный) = PC лучше",
              xlabel=f"Δ{metric}")

    colors = ["#27AE60" if v < 0 else "#E74C3C" for v in agg.values]
    ylabels = [ASSET_LABELS.get(t, t) for t in agg.index]
    bars = ax.barh(ylabels, agg.values, color=colors,
                   edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)

    max_abs = abs(agg.values).max() if len(agg) > 0 else 1
    for bar, val in zip(bars, agg.values):
        ax.text(val + np.sign(val) * max_abs * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center",
                ha="left" if val >= 0 else "right",
                color="black", fontsize=8)

    pc_wins  = (agg < 0).sum()
    exp_wins = (agg >= 0).sum()
    fig.text(0.98, 0.02,
             f"PC-DAG лучше: {pc_wins}/{len(agg)}\n"
             f"Expert-DAG лучше: {exp_wins}/{len(agg)}",
             ha="right", va="bottom", color="#AAAAAA", fontsize=9,
             transform=fig.transFigure)

    plt.tight_layout()
    path = FIGURES_DIR / f"dag_divergence_{metric}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    log.info(f"  ✓ {path.name}")
    plt.close()


def plot_causal_vs_baseline(full_df: pd.DataFrame, metric: str = "RMSE",
                            horizon: int = 1):
    """Grouped bar: лучший каузальный vs лучший baseline по активам."""
    sub = full_df[full_df["Horizon"] == horizon].copy()
    if sub.empty:
        return

    rows = []
    for tgt in sub["Target"].unique():
        tsub  = sub[sub["Target"] == tgt]
        c_val = tsub[tsub["Type"] == "Causal"][metric].dropna()
        b_val = tsub[tsub["Type"] == "Baseline"][metric].dropna()
        if c_val.empty or b_val.empty:
            continue
        rows.append({
            "label":   ASSET_LABELS.get(tgt, tgt),
            "causal":  c_val.min(),
            "baseline": b_val.min(),
            "group":   ASSET_GROUP.get(tgt, "A"),
        })

    if not rows:
        return

    df_plot = pd.DataFrame(rows)
    x = np.arange(len(df_plot))
    w = 0.38

    fig = _dark_fig(max(12, len(df_plot) * 1.3), 5.5)
    ax  = fig.add_subplot(111)
    _style_ax(ax,
              title=f"Лучший каузальный (любой Z) vs Лучший baseline — {metric}, h={horizon}",
              ylabel=metric)

    b1 = ax.bar(x - w / 2, df_plot["causal"],   w, label="Best Causal",
                color="#2196F3", edgecolor="black", linewidth=0.4, alpha=0.85)
    b2 = ax.bar(x + w / 2, df_plot["baseline"], w, label="Best Baseline",
                color="#FF9800", edgecolor="black", linewidth=0.4, alpha=0.85)

    for i, row in df_plot.iterrows():
        pct   = (row["baseline"] - row["causal"]) / (row["baseline"] + 1e-9) * 100
        color = "#27AE60" if pct > 0 else "#E74C3C"
        ypos  = max(row["causal"], row["baseline"]) * 1.03
        ax.text(x[i], ypos, f"{pct:+.1f}%",
                ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["label"], rotation=30, ha="right",
                       fontsize=9, color="black")
    ax.legend(facecolor=_CARD_BG, edgecolor=_GRID_CLR,
              labelcolor="white", fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / f"causal_vs_baseline_{metric}_h{horizon}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    log.info(f"  ✓ {path.name}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# КОНСОЛЬНЫЙ ОТЧЁТ
# ══════════════════════════════════════════════════════════════════════════════

def print_report(full_df: pd.DataFrame, dag_df: pd.DataFrame):
    SEP = "═" * 72

    print(f"\n{SEP}")
    print("  ИТОГОВЫЙ ОТЧЁТ: Каузальный пайплайн прогнозирования")
    print(SEP)

    # ── 1. Лучшие методы на h=1 ──────────────────────────────────────────────
    print("\n▸ Лучший метод по каждому активу (RMSE, h=1):")
    h1 = full_df[full_df["Horizon"] == 1]
    for tgt, grp in h1.groupby("Target"):
        valid = grp.dropna(subset=["RMSE"])
        if valid.empty:
            continue
        best = valid.loc[valid["RMSE"].idxmin()]
        ctrl = (f" [{best.get('ControlSet', '')}]"
                if best["Type"] == "Causal" else "")
        print(f"  {ASSET_LABELS.get(tgt, tgt):<12}  "
              f"{best['Method']:<18}{ctrl:<24}  RMSE={best['RMSE']:.4f}")

    # ── 2. Сравнение Expert vs PC DAG ────────────────────────────────────────
    if not dag_df.empty:
        by_asset = dag_df.groupby("Target")["diff_pc_minus_expert"].mean()
        pc_wins  = (by_asset < 0).sum()
        exp_wins = (by_asset >= 0).sum()
        print(f"\n▸ Expert-DAG vs PC-DAG (RMSE, среднее по горизонтам и методам):")
        print(f"   PC-DAG лучше     : {pc_wins}/{len(by_asset)} активов")
        print(f"   Expert-DAG лучше : {exp_wins}/{len(by_asset)} активов\n")
        for tgt, diff in by_asset.sort_values().items():
            winner = "← PC" if diff < 0 else "← Expert"
            print(f"   {ASSET_LABELS.get(tgt, tgt):<12}  Δ={diff:+.5f}  {winner}")

    # ── 3. Деградация по горизонтам ───────────────────────────────────────────
    print(f"\n▸ Средний RMSE по горизонтам прогноза:")
    horiz = (full_df.groupby(["Horizon", "Type"])["RMSE"]
             .mean().unstack("Type").round(4))
    print(horiz.to_string())

    # ── 4. По группам активов ─────────────────────────────────────────────────
    print(f"\n▸ Среднее RMSE по группам активов (h=1, лучший метод):")
    h1_best = (h1.groupby("Target")["RMSE"].min().reset_index())
    h1_best["Group"] = h1_best["Target"].map(ASSET_GROUP)
    grp_avg = h1_best.groupby("Group")["RMSE"].mean().round(4)
    group_names = {"A": "Энергетика/Металлы", "B": "Форекс",
                   "C": "Макро США",          "D": "Крипто"}
    for g, v in grp_avg.items():
        print(f"   Группа {g} ({group_names.get(g, g)}): {v:.4f}")

    print(f"\n{SEP}")
    print("  Файлы сохранены:")
    for p in sorted(RESULTS_DIR.glob("*.csv")):
        print(f"    results/{p.name}")
    for p in sorted(FIGURES_DIR.glob("*.png")):
        print(f"    figures/{p.name}")
    print(f"{SEP}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_comparison():
    log.info("Загружаем результаты step3 и step4...")
    causal, baseline = load_results()

    if not causal and not baseline:
        log.error("Результаты не найдены. Запустите step3 и step4 сначала.")
        return None, None

    log.info("Строим сводные таблицы...")
    full_df  = build_full_df(causal, baseline)
    summary  = build_summary_h1(full_df, metric="RMSE")
    dag_comp = build_dag_comparison(full_df, metric="RMSE")

    full_df.to_csv(RESULTS_DIR  / "full_comparison.csv", index=False)
    summary.to_csv(RESULTS_DIR  / "summary_h1_RMSE.csv")
    dag_comp.to_csv(RESULTS_DIR / "dag_comparison.csv", index=False)
    log.info("  ✓ CSV-таблицы сохранены")

    horizons = sorted(full_df["Horizon"].dropna().unique().astype(int).tolist())

    log.info("Генерируем графики...")
    for h in horizons:
        plot_heatmap_by_horizon(full_df, metric="RMSE", horizon=h)
        plot_causal_vs_baseline(full_df, metric="RMSE", horizon=h)

    plot_horizon_progression(full_df, metric="RMSE")
    plot_dag_divergence(dag_comp, metric="RMSE")

    print_report(full_df, dag_comp)
    return full_df, dag_comp


if __name__ == "__main__":
    run_comparison()
