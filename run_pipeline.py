"""
causal_pipeline/run_pipeline.py
════════════════════════════════
Главный скрипт: запускает все шаги каузального пайплайна.

Использование:
  # Полный запуск (все 10 активов, все методы)
  python run_pipeline.py

  # Запуск только одного шага
  python run_pipeline.py --step 2
  python run_pipeline.py --step 3
  python run_pipeline.py --step 5

  # Запуск только для конкретных активов
  python run_pipeline.py --assets WTI_oil BTC ETH

  # Без PC-алгоритма (только экспертный DAG)
  python run_pipeline.py --no-pcmci              # без PCMCI+, только expert DAG

  # Горизонт прогноза
  python run_pipeline.py --horizon 5

Зависимости:
  pip install causallearn econml linearmodels pmdarima prophet

              torch scikit-learn statsmodels pandas numpy matplotlib
              xgboost shap
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/pipeline.log", mode="w", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# Создаём папки
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))


# ══════════════════════════════════════════════════════════════════════════════
# ПРОВЕРКА ЗАВИСИМОСТЕЙ
# ══════════════════════════════════════════════════════════════════════════════

def check_dependencies():
    """Проверяет наличие ключевых пакетов."""
    packages = {
        "numpy":         "numpy",
        "pandas":        "pandas",
        "scipy":         "scipy",
        "statsmodels":   "statsmodels",
        "sklearn":       "scikit-learn",
        "matplotlib":    "matplotlib",
    }
    optional = {
        "causallearn":   "causallearn",
        "econml":        "econml",
        "linearmodels":  "linearmodels",
        "pmdarima":      "pmdarima",
        "prophet":       "prophet",
        "torch":         "torch",
        "tensorflow":    "tensorflow",
        "xgboost":       "xgboost",
    }

    log.info("Проверка зависимостей:")
    missing_required = []
    for mod, pkg in packages.items():
        try:
            __import__(mod)
            log.info(f"  ✓ {pkg}")
        except ImportError:
            log.error(f"  ✗ {pkg}  ← ТРЕБУЕТСЯ")
            missing_required.append(pkg)

    log.info("Опциональные пакеты:")
    for mod, pkg in optional.items():
        try:
            __import__(mod)
            log.info(f"  ✓ {pkg}")
        except ImportError:
            log.warning(f"  ○ {pkg}  (пропускается, будет использован fallback)")

    if missing_required:
        log.error(f"\nУстановите обязательные пакеты:\n"
                  f"  pip install {' '.join(missing_required)}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# ШАГИ ПАЙПЛАЙНА
# ══════════════════════════════════════════════════════════════════════════════

def step1_info():
    """Шаг 1: показывает загруженные DAG-объекты."""
    log.info("\n" + "═"*55)
    log.info("  ШАГ 1: Экспертные DAG и определение причин")
    log.info("═"*55)
    from step1_dag_definitions import ALL_DAGS
    for k, dag in ALL_DAGS.items():
        log.info(f"  {k:12s} │ {dag.group} │ freq={dag.freq} │ "
                 f"causes={len(dag.causes)} │ confounders={len(dag.confounders)}")
    return ALL_DAGS


def step2_select(use_pcmci: bool = True, assets: list = None) -> dict:
    """Шаг 2: отбор переменных."""
    log.info("\n" + "═"*55)
    log.info("  ШАГ 2: Отбор контрольных переменных")
    log.info("═"*55)
    from step2_variable_selection import run_variable_selection, ALL_DAGS

    var_sel = run_variable_selection(use_pcmci=use_pcmci)

    # Фильтрация по активам
    if assets:
        var_sel = {k: v for k, v in var_sel.items() if k in assets}
    return var_sel


def step3_causal(var_sel: dict) -> dict:
    """Шаг 3: каузальные модели."""
    log.info("\n" + "═"*55)
    log.info("  ШАГ 3: Каузальные модели прогнозирования")
    log.info("═"*55)
    from step3_causal_models import run_all_causal_models
    return run_all_causal_models(var_sel)


def step4_baseline(var_sel: dict) -> dict:
    """Шаг 4: baseline модели."""
    log.info("\n" + "═"*55)
    log.info("  ШАГ 4: Baseline модели (без каузальной коррекции)")
    log.info("═"*55)
    from step4_baseline_models import run_all_baselines
    return run_all_baselines(var_sel)


def step5_compare():
    """Шаг 5: сравнение и отчёт."""
    log.info("\n" + "═"*55)
    log.info("  ШАГ 5: Сравнение результатов и отчёт")
    log.info("═"*55)
    from step5_comparison_report import run_comparison
    return run_comparison()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Каузальный пайплайн прогнозирования финансовых рядов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры запуска:
  python run_pipeline.py                          # полный пайплайн, все 10 активов
  python run_pipeline.py --assets WTI_oil BTC     # только выбранные активы
  python run_pipeline.py --step 5                 # только отчёт (если step3/4 уже готовы)
  python run_pipeline.py --no-pcmci               # без PCMCI+, только expert DAG
  python run_pipeline.py --assets CPI UMCSENT IndProd  # только макро-группа C

Адаптивные горизонты прогноза (задаются автоматически по частоте):
  Дневные (A/B/D): h=1, 5, 21 (1 день, 1 неделя, 1 месяц)
  Месячные (C):   h=1, 3, 6  (1 мес, квартал, полгода)

DAG-подходы для контрольных переменных Z:
  heuristic     — корреляционный отбор (baseline)
  expert        — экспертный backdoor criterion (step1_dag_definitions.py)
  pc            — автоматический PC-алгоритм (causallearn)
  → Все три прогоняются и сравниваются в step5.
        """
    )
    parser.add_argument(
        "--step", type=int, default=0,
        help="Запустить конкретный шаг (1-5). 0 = весь пайплайн."
    )
    parser.add_argument(
        "--assets", nargs="+", default=None,
        help="Список активов: WTI_oil NatGas Gold EURUSD GBPUSD USDJPY "
             "CPI IndProd UMCSENT BTC ETH"
    )
    parser.add_argument(
        "--no-pcmci", action="store_true",
        help="Отключить PC-алгоритм (использовать только экспертный DAG)"
    )
    parser.add_argument(
        "--skip-deps-check", action="store_true",
        help="Пропустить проверку зависимостей"
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t_start = time.time()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   Каузальный пайплайн прогнозирования финансовых     ║")
    log.info("║   рядов: 10 активов × 4 каузальных × 4 baseline      ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    if not args.skip_deps_check:
        check_dependencies()

    use_pcmci = not getattr(args, "no_pcmci", False)

    sel_path = Path("results/step2_variable_selection.json")
    var_sel  = None

    # ── Запуск по шагам ───────────────────────────────────────────────────────
    if args.step == 0 or args.step == 1:
        step1_info()

    if args.step == 0 or args.step == 2:
        var_sel = step2_select(use_pcmci=use_pcmci, assets=args.assets)
    elif sel_path.exists():
        with open(sel_path) as f:
            var_sel = json.load(f)
        if args.assets:
            var_sel = {k: v for k, v in var_sel.items() if k in args.assets}
    else:
        log.warning("step2 не выполнен — запускаем автоматически")
        var_sel = step2_select(use_pcmci=use_pcmci, assets=args.assets)

    if args.step == 0 or args.step == 3:
        step3_causal(var_sel)

    if args.step == 0 or args.step == 4:
        step4_baseline(var_sel)

    if args.step == 0 or args.step == 5:
        step5_compare()

    elapsed = time.time() - t_start
    log.info(f"\n✓ Пайплайн завершён за {elapsed:.1f} сек ({elapsed/60:.1f} мин)")
    log.info(f"  Результаты:  results/")
    log.info(f"  Графики:     figures/")


if __name__ == "__main__":
    main()
