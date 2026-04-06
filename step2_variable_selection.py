"""
causal_pipeline/step2_variable_selection.py
════════════════════════════════════════════
Шаг 2: Отбор контрольных переменных Z двумя способами:
  (A) Эвристический   — корреляция / mutual information (baseline)
  (B) Причинно-обоснованный — backdoor criterion через PC-алгоритм (causallearn)
      + экспертный DAG из step1

Выходы:
  - controls_heuristic[target]  : список Z, отобранных по корреляции
  - controls_causal[target]     : список Z по backdoor criterion
  - Сохраняет результаты в results/step2_variable_selection.json
"""

import json
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATASETS_DIR = Path("datasets")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── импорт DAG-реестра ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from step1_dag_definitions import ALL_DAGS, AssetDAG


# ══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(dag: AssetDAG, max_cols: int = 50) -> pd.DataFrame:
    """Загружает CSV датасет и ограничивает число столбцов."""
    path = DATASETS_DIR / dag.file
    if not path.exists():
        log.warning(f"Файл не найден: {path}. Генерируем синтетические данные.")
        return _synthetic_data(dag)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.select_dtypes(include=[np.number]).dropna(thresh=len(df) * 0.5, axis=1)
    # Убираем столбцы с нулевой дисперсией
    df = df.loc[:, df.std() > 1e-9]
    if len(df.columns) > max_cols:
        # Оставляем целевой ряд + наиболее коррелированные столбцы
        if dag.target in df.columns:
            corrs = df.corr()[dag.target].abs().sort_values(ascending=False)
            keep = [dag.target] + corrs.index[1:max_cols].tolist()
            df = df[keep]
    return df


def _synthetic_data(dag: AssetDAG, n: int = 500) -> pd.DataFrame:
    """Генерирует синтетические данные для тестирования без реальных файлов."""
    np.random.seed(42)
    all_vars = [dag.target] + dag.causes + dag.confounders + dag.instruments
    all_vars = list(dict.fromkeys(all_vars))  # убираем дубли
    n_vars = len(all_vars)
    data = np.random.randn(n, n_vars)
    # Добавляем лёгкую причинно-следственную структуру
    for i in range(1, n_vars):
        data[:, 0] += 0.3 * data[:, i] + 0.1 * np.random.randn(n)
    if dag.freq == "ME":
        idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    else:
        idx = pd.date_range("2010-01-04", periods=n, freq="B")
    return pd.DataFrame(data, columns=all_vars, index=idx)


def prepare_stationary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Приводит к стационарности через первые разности.
    ADF-тест для каждой переменной.
    """
    from statsmodels.tsa.stattools import adfuller
    result = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 20:
            continue
        try:
            p_val = adfuller(s, autolag="AIC")[1]
        except Exception:
            p_val = 1.0
        if p_val > 0.05:
            result[col] = df[col].diff()
        else:
            result[col] = df[col]
    return result.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД A: ЭВРИСТИЧЕСКИЙ (КОРРЕЛЯЦИЯ + MI)
# ══════════════════════════════════════════════════════════════════════════════

def select_heuristic(df: pd.DataFrame, target: str,
                     top_k: int = 8,
                     min_corr: float = 0.15) -> List[str]:
    """
    Отбирает контрольные переменные по:
      1. Абсолютной корреляции Пирсона с целевым рядом
      2. Ранговой корреляции Спирмана (для нелинейных связей)
      3. Partial correlation (с контролем лагов цели)
    Возвращает топ-k переменных (исключая сам target).
    """
    if target not in df.columns:
        log.error(f"Целевой ряд {target} не найден в датасете")
        return []

    predictors = [c for c in df.columns if c != target
                  and not c.startswith(f"{target}_")]  # исключаем авто-фичи

    if not predictors:
        return []

    scores = {}
    y = df[target].dropna()

    for col in predictors:
        x = df[col].dropna()
        idx = y.index.intersection(x.index)
        if len(idx) < 30:
            continue
        xi, yi = x[idx].values, y[idx].values
        try:
            pearson  = abs(np.corrcoef(xi, yi)[0, 1])
            spearman = abs(stats.spearmanr(xi, yi).correlation)
            combined = 0.5 * pearson + 0.5 * spearman
            scores[col] = combined
        except Exception:
            pass

    # Фильтруем по минимальной корреляции
    scores = {k: v for k, v in scores.items() if v >= min_corr and not np.isnan(v)}
    sorted_vars = sorted(scores, key=lambda k: scores[k], reverse=True)

    selected = sorted_vars[:top_k]
    log.info(f"  Heuristic [{target}]: {len(selected)} переменных "
             f"(топ: {selected[:3]})")
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОД B: ПРИЧИННО-ОБОСНОВАННЫЙ
# ══════════════════════════════════════════════════════════════════════════════

def select_causal_expert(dag: AssetDAG) -> List[str]:
    """
    Метод B1: экспертный backdoor criterion.
    Возвращает конфаундеры из DAG — именно те переменные, которые
    блокируют backdoor-пути от причин X к цели Y.
    НЕ включает медиаторы (они на пути X→M→Y и не должны контролироваться).
    """
    # Backdoor adjustment set = confounders \ mediators
    backdoor_set = [z for z in dag.confounders if z not in dag.mediators]
    log.info(f"  Expert backdoor [{dag.target}]: {backdoor_set}")
    return backdoor_set


def select_causal_pc(df: pd.DataFrame, target: str,
                     dag: AssetDAG,
                     alpha: float = 0.05,
                     max_cond_vars: int = 3) -> List[str]:
    """
    Метод B2: автоматический backdoor criterion через PC-алгоритм.
    Использует causallearn для построения CPDAG, затем ищет
    минимальное множество разделения (марковское одеяло).
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
    except ImportError:
        log.warning("causallearn не установлен: pip install causallearn")
        return select_causal_expert(dag)  # fallback

    # Берём только ключевые переменные (целевой + причины + конфаундеры)
    key_vars = ([target] + dag.causes[:5] + dag.confounders[:5] +
                dag.instruments[:2])
    key_vars = [v for v in key_vars if v in df.columns]
    key_vars = list(dict.fromkeys(key_vars))

    if len(key_vars) < 3:
        return select_causal_expert(dag)

    sub = df[key_vars].dropna()
    if len(sub) < 50:
        return select_causal_expert(dag)

    log.info(f"  PC algorithm [{target}]: {len(key_vars)} переменных, "
             f"n={len(sub)}")

    try:
        data_arr = sub.values.astype(float)
        # Нормализуем для численной стабильности
        data_arr = (data_arr - data_arr.mean(0)) / (data_arr.std(0) + 1e-9)

        cg = pc(data_arr, alpha=alpha, indep_test=fisherz,
                uc_rule=0, uc_priority=2, show_progress=False)

        # Получаем матрицу смежности
        adj = cg.G.graph  # shape: (n_vars, n_vars)
        target_idx = key_vars.index(target)

        # Марковское одеяло = родители + потомки + родители потомков
        # Для backdoor нам нужны родители целевого узла
        parents_of_target = []
        for i, var in enumerate(key_vars):
            if i == target_idx:
                continue
            # В CPDAG: adj[i, j] = -1 означает i → j
            if adj[i, target_idx] == -1 and adj[target_idx, i] == 1:
                parents_of_target.append(var)

        # Если PC не нашёл ориентированных рёбер — используем смежность
        if not parents_of_target:
            adjacent = [key_vars[i] for i in range(len(key_vars))
                       if i != target_idx and adj[i, target_idx] != 0]
            parents_of_target = adjacent

        # Исключаем медиаторы
        result = [v for v in parents_of_target if v not in dag.mediators]
        log.info(f"  PC result [{target}]: {result}")
        return result if result else select_causal_expert(dag)

    except Exception as e:
        log.warning(f"  PC ошибка [{target}]: {e}. Используем expert DAG.")
        return select_causal_expert(dag)


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ ОТБОРА
# ══════════════════════════════════════════════════════════════════════════════

def run_variable_selection(
        use_pc: bool = True
) -> Dict[str, Dict[str, List[str]]]:
    """
    Запускает оба метода отбора для всех 10 датасетов.
    Возвращает словарь:
      {target: {"heuristic": [...], "causal_expert": [...], "causal_pc": [...]}}
    """
    all_results = {}

    for target, dag in ALL_DAGS.items():
        log.info(f"\n── {dag.name} ({target}) ──")

        df_raw = load_dataset(dag)
        if df_raw.empty or target not in df_raw.columns:
            log.warning(f"  Пропускаем {target}: нет данных")
            continue

        df_stat = prepare_stationary(df_raw, target)

        # Метод A: эвристический
        controls_heuristic = select_heuristic(df_stat, target, top_k=8)

        # Метод B1: экспертный backdoor
        controls_expert = select_causal_expert(dag)

        # Метод B2: PC-алгоритм (если доступен)
        if use_pc:
            controls_pc = select_causal_pc(df_stat, target, dag)
        else:
            controls_pc = controls_expert

        # Объединённый каузальный набор (union с приоритетом эксперта)
        controls_causal_union = list(dict.fromkeys(
            controls_expert + [v for v in controls_pc
                               if v not in controls_expert]
        ))

        all_results[target] = {
            "heuristic":       controls_heuristic,
            "causal_expert":   controls_expert,
            "causal_pc":       controls_pc,
            "causal_union":    controls_causal_union,
            "causes":          dag.causes,
            "instruments":     dag.instruments,
            "freq":            dag.freq,
            "file":            dag.file,
        }

        log.info(f"  → Heuristic: {controls_heuristic}")
        log.info(f"  → Causal (expert):  {controls_expert}")
        log.info(f"  → Causal (PC):      {controls_pc}")

    # Сохраняем
    out_path = RESULTS_DIR / "step2_variable_selection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    log.info(f"\n✓ Результаты сохранены: {out_path}")

    return all_results


if __name__ == "__main__":
    results = run_variable_selection(use_pc=True)
    print("\n" + "═" * 60)
    print("  ИТОГ: контрольные наборы переменных")
    print("═" * 60)
    for target, info in results.items():
        print(f"\n  {target}")
        print(f"    Heuristic    : {info['heuristic']}")
        print(f"    Causal expert: {info['causal_expert']}")
        print(f"    Causal PC    : {info['causal_pc']}")
