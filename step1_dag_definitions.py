"""
causal_pipeline/step1_dag_definitions.py
════════════════════════════════════════
Шаг 1–2: Экспертные DAG + backdoor criterion для каждого целевого ряда.
Определяет:
  - causes      : гипотетические причины X (прямые)
  - confounders : конфаундеры Z (общие причины X→Y)
  - controls_heuristic  : контроли по корреляции (baseline)
  - controls_causal     : контроли по backdoor criterion
  - instruments : инструментальные переменные (для IV/DML)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class AssetDAG:
    """Хранит экспертный DAG и наборы переменных для одного целевого ряда."""
    name: str
    target: str
    group: str          # A / B / C / D
    freq: str           # 'D' = daily, 'ME' = monthly
    file: str           # имя CSV из datasets/

    # Гипотетические причины X (шаг 1)
    causes: List[str] = field(default_factory=list)

    # Конфаундеры Z (открывают backdoor-пути X → Y)
    confounders: List[str] = field(default_factory=list)

    # Медиаторы (переменные на пути X → M → Y — НЕ контролируем)
    mediators: List[str] = field(default_factory=list)

    # Инструменты (влияют на X, но не на Y напрямую)
    instruments: List[str] = field(default_factory=list)

    # Описание backdoor-путей (для документации)
    backdoor_paths: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# ГРУППА A — Энергетика и Металлы
# ══════════════════════════════════════════════════════════════════════════════

WTI_DAG = AssetDAG(
    name="WTI Crude Oil",
    target="WTI_oil",
    group="A",
    freq="D",
    file="A_WTI_oil.csv",
    causes=[
        "EIA_crude_stocks",    # запасы нефти США — прямая причина цены
        "Brent",               # арбитраж WTI–Brent
        "DXY",                 # нефть котируется в USD
        "ISM_PMI",             # промышленный спрос
        "OPEC_proxy",          # предложение (прокси через Brent spread)
    ],
    confounders=[
        "VIX",                 # общий риск-фактор → DXY и WTI
        "SP500",               # risk-on/off → DXY, WTI, спрос
        "Yield10Y",            # ставки → DXY → WTI
        "IndProd_USA",         # производство → спрос → WTI
    ],
    mediators=[
        "WTI_Brent_spread",    # медиатор между OPEC и WTI (не контролируем)
    ],
    instruments=[
        "EIA_gas_stocks",      # влияет на энергокомплекс, но не на WTI напрямую
    ],
    backdoor_paths=[
        "VIX → DXY → WTI_oil",
        "SP500 → risk_appetite → DXY → WTI_oil",
        "Yield10Y → DXY → WTI_oil",
        "IndProd_USA → demand → WTI_oil",
    ]
)

NATGAS_DAG = AssetDAG(
    name="Natural Gas (Henry Hub)",
    target="NatGas",
    group="A",
    freq="D",
    file="A_NatGas.csv",
    causes=[
        "EIA_gas_stocks",      # запасы газа — ключевой драйвер
        "WTI_oil",             # энергетический переключатель (fuel switching)
        "Copper",              # промышленный спрос (прокси)
        "ISM_PMI",             # производственный спрос
    ],
    confounders=[
        "VIX",
        "SP500",
        "DXY",
        "IndProd_USA",
    ],
    instruments=[
        "EIA_crude_stocks",    # коррелирует с энергокомплексом, не с газом напрямую
    ],
    backdoor_paths=[
        "VIX → risk_off → WTI → NatGas",
        "IndProd_USA → energy_demand → NatGas",
        "DXY → commodity_prices → NatGas",
    ]
)

GOLD_DAG = AssetDAG(
    name="Gold",
    target="Gold",
    group="A",
    freq="D",
    file="A_Gold.csv",
    causes=[
        "DXY",                 # золото — альтернатива USD
        "Yield10Y",            # реальная ставка → opportunity cost золота
        "VIX",                 # safe-haven спрос
        "InflationCPI",        # инфляционный хедж
        "Silver",              # металлы-компаньоны
    ],
    confounders=[
        "SP500",               # risk-on/off → Gold и VIX
        "IndProd_USA",         # экономический цикл → инфляция и Gold
        "FedFunds",            # ставка ФРС → Yield10Y → Gold
    ],
    instruments=[
        "Copper",              # промышленный металл, не safe-haven
    ],
    backdoor_paths=[
        "SP500 → risk_off → VIX → Gold",
        "FedFunds → Yield10Y → real_rate → Gold",
        "IndProd_USA → inflation → Gold",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# ГРУППА B — Валютный рынок
# ══════════════════════════════════════════════════════════════════════════════

EURUSD_DAG = AssetDAG(
    name="EUR/USD",
    target="EURUSD",
    group="B",
    freq="D",
    file="B_EURUSD.csv",
    causes=[
        "T10Y2Y",              # кривая доходности США (carry)
        "UST_10Y",             # ставки США
        "FedFunds",            # монетарная политика ФРС
        "Trade_Balance",       # текущий счёт → спрос на EUR
        "M2_USA",              # денежная масса → инфляция → курс
    ],
    confounders=[
        "VIX",                 # глобальный риск → бегство в USD
        "SP500",               # risk-on → ослабление USD
        "Gold",                # safe-haven вместе с EUR
        "WTI",                 # нефть → торговый баланс Европы
    ],
    instruments=[
        "UST_2Y",              # краткосрочные ставки США (быстрее реагируют на ФРС)
    ],
    backdoor_paths=[
        "VIX → USD_safe_haven_demand → EURUSD",
        "SP500 → risk_on → EURUSD",
        "WTI → EU_trade_balance → EURUSD",
    ]
)

GBPUSD_DAG = AssetDAG(
    name="GBP/USD",
    target="GBPUSD",
    group="B",
    freq="D",
    file="B_GBPUSD.csv",
    causes=[
        "FedFunds",
        "T10Y2Y",
        "Trade_Balance",
        "M2_USA",
        "EURUSD",              # EUR/GBP кросс
    ],
    confounders=[
        "VIX",
        "SP500",
        "Gold",
        "WTI",
    ],
    instruments=["UST_2Y"],
    backdoor_paths=[
        "VIX → USD_safe_haven → GBPUSD",
        "EURUSD → EUR_GBP_cross → GBPUSD",
    ]
)

USDJPY_DAG = AssetDAG(
    name="USD/JPY",
    target="USDJPY",
    group="B",
    freq="D",
    file="B_USDJPY.csv",
    causes=[
        "Yield10Y",            # дифференциал ставок США–Япония (главный драйвер)
        "UST_10Y",
        "FedFunds",
        "VIX",                 # JPY — safe-haven актив
        "SP500",               # risk-on ослабляет JPY
    ],
    confounders=[
        "Gold",                # конкурирующий safe-haven
        "WTI",                 # импорт нефти Японии → JPY
        "DXY",                 # общая сила USD
    ],
    instruments=[
        "UST_2Y",
        "T10Y2Y",
    ],
    backdoor_paths=[
        "VIX → JPY_safe_haven → USDJPY",
        "SP500 → risk_on → carry_trade → USDJPY",
        "WTI → Japan_trade_deficit → JPY_weakening → USDJPY",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# ГРУППА C — Макро-индикаторы США (месячная частота)
# ══════════════════════════════════════════════════════════════════════════════

CPI_DAG = AssetDAG(
    name="US CPI Inflation",
    target="CPI",
    group="C",
    freq="ME",
    file="C_CPI.csv",
    causes=[
        "PPI",                 # производственные цены передаются в CPI
        "WTI_oil",             # энергетическая составляющая CPI
        "M2",                  # количественная теория денег
        "NFP",                 # напряжённость рынка труда → зарплаты → CPI
        "RetailSales",         # спрос → ценовое давление
    ],
    confounders=[
        "FedFunds",            # ставка ФРС → кредит → спрос → CPI
        "Unemployment",        # рынок труда → зарплаты (Phillips curve)
        "HousingStarts",       # жильё → CPI компонент
    ],
    mediators=[
        "PCE_core",            # медиатор PPI → PCE → CPI (не контролируем)
    ],
    instruments=[
        "EIA_crude_stocks",    # запасы нефти → нефтяные цены → CPI (не зависит от спроса)
    ],
    backdoor_paths=[
        "FedFunds → credit_conditions → demand → CPI",
        "Unemployment → wage_growth → CPI",
        "HousingStarts → housing_supply → rent_CPI",
    ]
)

INDPROD_DAG = AssetDAG(
    name="Industrial Production",
    target="IndProd",
    group="C",
    freq="ME",
    file="C_IndProd.csv",
    causes=[
        "ISM_PMI",             # PMI — leading indicator IP
        "NFP",                 # занятость в производстве
        "RetailSales",         # спрос → производство
        "WTI_oil",             # энергозатраты производства
        "FedFunds",            # стоимость капитала → инвестиции → IP
    ],
    confounders=[
        "Unemployment",
        "HousingStarts",       # строительный сектор → IP
        "M2",                  # ликвидность → инвестиции
    ],
    instruments=[
        "EIA_crude_stocks",    # предложение энергии (не зависит от IP напрямую)
    ],
    backdoor_paths=[
        "M2 → credit → investment → IndProd",
        "Unemployment → labour_supply → IndProd",
        "FedFunds → borrowing_cost → capex → IndProd",
    ]
)

UMCSENT_DAG = AssetDAG(
    name="Michigan Consumer Sentiment",
    target="UMCSENT",
    group="C",
    freq="ME",
    file="C_UMCSENT.csv",
    causes=[
        "Unemployment",        # безработица напрямую влияет на настроения
        "CPI",                 # инфляция снижает покупательную способность
        "RetailSales",         # потребительская активность
        "SP500",               # "wealth effect" → настроения
        "WTI_oil",             # цена бензина → видимая инфляция
    ],
    confounders=[
        "NFP",                 # занятость → безработица и настроения
        "FedFunds",            # ставки → ипотека → настроения
        "Savings_Rate",        # финансовая подушка → оптимизм
        "CredCardDebt",        # долговая нагрузка → пессимизм
    ],
    instruments=[
        "EIA_gas_stocks",      # запасы газа → цены энергоносителей (экзогенно)
    ],
    backdoor_paths=[
        "NFP → income_security → UMCSENT",
        "FedFunds → mortgage_rate → housing_affordability → UMCSENT",
        "Savings_Rate → financial_buffer → UMCSENT",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# ГРУППА D — Цифровые активы
# ══════════════════════════════════════════════════════════════════════════════

BTC_DAG = AssetDAG(
    name="Bitcoin",
    target="BTC",
    group="D",
    freq="D",
    file="D_BTC.csv",
    causes=[
        "M2",                  # глобальная ликвидность → крипто
        "DXY",                 # инверсия: слабый USD → BTC рост
        "SP500",               # risk-on → BTC
        "ETH",                 # ETH как ведомый актив (или опережает)
        "BTC_Volume",          # объём как прокси спроса/давления
        "FedFunds",            # ставки → риск-аппетит → BTC
    ],
    confounders=[
        "VIX",                 # общий риск → SP500, DXY, BTC
        "Gold",                # конкурирующий «цифровой хедж»
        "Yield10Y",            # реальные ставки → альтернативные активы
    ],
    mediators=[
        "BTC_ETH_ratio",       # медиатор (доминирование) — не контролируем
    ],
    instruments=[
        "BTC_DailyRange",      # внутридневная волатильность — прокси ончейн-активности
    ],
    backdoor_paths=[
        "VIX → risk_off → SP500_down → BTC_down",
        "VIX → USD_safe_haven → DXY_up → BTC_down",
        "FedFunds → real_yield → opportunity_cost → BTC",
        "Yield10Y → discount_rate → speculative_assets → BTC",
    ]
)

ETH_DAG = AssetDAG(
    name="Ethereum",
    target="ETH",
    group="D",
    freq="D",
    file="D_ETH.csv",
    causes=[
        "BTC",                 # BTC ведёт за собой весь рынок
        "M2",
        "DXY",
        "SP500",
        "ETH_Volume",
        "SOL",                 # конкурент за DeFi/NFT рынок
    ],
    confounders=[
        "VIX",
        "Gold",
        "Yield10Y",
        "FedFunds",
    ],
    instruments=[
        "ETH_DailyRange",
        "BTC_DailyRange",      # волатильность BTC влияет на ETH через рынок
    ],
    backdoor_paths=[
        "VIX → risk_off → BTC_down → ETH_down",
        "FedFunds → risk_appetite → BTC → ETH",
        "M2 → global_liquidity → crypto_market → ETH",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# РЕЕСТР: все DAG-объекты
# ══════════════════════════════════════════════════════════════════════════════

ALL_DAGS: Dict[str, AssetDAG] = {
    "WTI_oil":  WTI_DAG,
    "NatGas":   NATGAS_DAG,
    "Gold":     GOLD_DAG,
    "EURUSD":   EURUSD_DAG,
    "GBPUSD":   GBPUSD_DAG,
    "USDJPY":   USDJPY_DAG,
    "CPI":      CPI_DAG,
    "IndProd":  INDPROD_DAG,
    "UMCSENT":  UMCSENT_DAG,
    "BTC":      BTC_DAG,
    "ETH":      ETH_DAG,
}

if __name__ == "__main__":
    print(f"Загружено DAG-объектов: {len(ALL_DAGS)}")
    for k, dag in ALL_DAGS.items():
        print(f"  {k:12s} | группа {dag.group} | freq={dag.freq} | "
              f"причин={len(dag.causes)} | конфаунд={len(dag.confounders)}")
