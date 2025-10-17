DOMAIN_KNOWLEDGE = {
    "tariff_entities": {
        "description": "Specific generator entities present in the tariff_with_usd view, used for tariff-based correlation and chart labeling.",
        "hydro": {
            "engurhesi": {
                "entities": ['ltd \"engurhesi\"1'],
                "labels": {"ltd \"engurhesi\"1": "Enguri HPP"}
            },
            "other_hydro": {
                "entities": [
                    'jsc \"energo-pro georgia genration\" (dzevrulhesi)',
                    'jsc \"energo-pro georgia genration\" (gumathesi)',
                    'jsc \"energo-pro georgia genration\" (shaorhesi)',
                    'jsc \"energo-pro georgia genration\" (rionhesi)',
                    'jsc \"energo-pro georgia genration\" (lajanurhesi)',
                    'jsc \"georgian water & power\" (zhinvalhesi)',
                    'ltd \"vardnili hpp cascade\"',
                    'ltd \"vartsikhe-2005\"',
                    'ltd \"khrami_1\"',
                    'ltd \"khrami_2\"'
                ],
                "labels": {
                    'jsc \"energo-pro georgia genration\" (dzevrulhesi)': "Dzevruli HPP",
                    'jsc \"energo-pro georgia genration\" (gumathesi)': "Gumati HPP",
                    'jsc \"energo-pro georgia genration\" (shaorhesi)': "Shaori HPP",
                    'jsc \"energo-pro georgia genration\" (rionhesi)': "Rioni HPP",
                    'jsc \"energo-pro georgia genration\" (lajanurhesi)': "Lajanuri HPP",
                    'jsc \"georgian water & power\" (zhinvalhesi)': "Zhinvali HPP",
                    'ltd \"vardnili hpp cascade\"': "Vardnili HPP Cascade",
                    'ltd \"vartsikhe-2005\"': "Vartsikhe HPP",
                    'ltd \"khrami_1\"': "Khrami I HPP",
                    'ltd \"khrami_2\"': "Khrami II HPP"
                }
            }
        },
        "thermal": {
            "entities": [
                'ltd \"gardabni thermal power plant\"',
                'ltd \"mtkvari energy\"',
                'ltd \"iec\" (tbilresi)',
                'ltd \"g power\" (capital turbines)'
            ],
            "labels": {
                'ltd \"gardabni thermal power plant\"': "Gardabani TPP",
                'ltd \"mtkvari energy\"': "Mtkvari Energy",
                'ltd \"iec\" (tbilresi)': "Tbilisi TPP",
                'ltd \"g power\" (capital turbines)': "G-POWER"
            }
        },
        "notes": [
            "Engurhesi is Georgia's main large hydro plant; used as a reference for hydro-tariff correlation.",
            "Thermal tariffs depend strongly on natural gas prices.",
            "'ltd \"mtkvari energy\"', 'ltd \"iec\" (tbilresi)', 'ltd \"g power\" (capital turbines)' are regulated old TPPs.",
            "'ltd \"gardabni thermal power plant\"' is a regulated new TPP.",
            "Energo-Pro hydro plants (Rioni, Lajanuri, Shaori, Gumati, Dzevruli) have similar cost structures.",
            "Entity labels are provided for clearer chart legends and report outputs."
        ]
    },

    "price_with_usd": {
        "description": "Electricity market prices (including balancing electricity) converted to USD.",
        "dependencies": [
            "Balancing electricity price represents the weighted average price of all trades executed for balancing purposes.",
            "It depends on the volumes and prices of electricity sold as balancing energy.",
            "Deregulated hydro and regulated HPP are typically the cheapest sources; regulated_new_TPP and thermal_PPA mid-level; import most expensive.",
            "Renewable_PPA drives balancing electricity price, particularly in summer.",
            "Balancing prices in summer are lower due to high hydro generation."
        ]
    },

    "trade": {
        "description": "Monthly trading data from market participants.",
        "dependencies": [
            "Includes transactions across exchange and balancing segments. The Exchange was introduced in July 2024.",
            "Trade volumes determine the weights used in calculating the balancing electricity price."
        ]
    },

    "CurrencyInfluence": {
        "GEL_USD_Effect": "Electricity price analysis must consider GEL/USD exchange rate effects.",
        "USD_Denominated_Costs": [
            "Natural gas for thermal generation is priced in USD.",
            "Electricity imports are priced in USD.",
            "Regulated tariffs are adjusted for exchange-rate changes to cover USD-denominated costs."
        ],
        "SeasonalityHint": "Compare yearly averages to neutralize strong seasonal swings (hydro vs thermal)."
    },

    "PriceDrivers": {
        "TargetColumns": ["p_bal_gel", "p_bal_usd"],
        "DriverColumns": {
            "p_dereg_gel": "Deregulated HPP Price",
            "p_gardabani_tpp_tariff": "Regulated New TPP Tariff",
            "p_grouped_old_tpp_tariffs": "Regulated Old TPP Group Tariffs",
            "entity = 'deregulated_hydro'": "Deregulated HPP Share",
            "entity = 'import'": "Import Share",
            "entity = 'renewable_ppa'": "Renewable PPA Share"
        }
    },

    "BalancingPriceFormation": {
        "Definition": "The balancing price is the weighted-average price of electricity sold on the balancing market.",
        "WeightingEntities": [
            "deregulated_hydro", "import", "regulated_hpp",
            "regulated_new_tpp", "regulated_old_tpp",
            "renewable_ppa", "thermal_ppa"
        ],
        "CalculationRule": "Weights are based on electricity sold as balancing energy by each entity. Total balancing quantity = sum of all listed entities from trade_derived_entities.",
        "AnalyticalUse": [
            "Compute average balancing price weighted by balancing-market quantities.",
            "Compare composition changes (hydro vs thermal/import) seasonally."
        ],
        "Insight": "Prices rise in winter when thermal/import shares grow, and fall in summer when hydro dominates."
    },

    "TariffStructure": {
        "Methodology": "Tariffs are approved by GNERC using cost-plus methodology.",
        "Components": {
            "Hydro": "Mainly fixed O&M and depreciation; minimal variable costs.",
            "Thermal": {
                "FixedComponent": "Guaranteed Capacity Fee (covers fixed costs).",
                "VariableComponent": "Per-MWh fee depends on gas price and efficiency.",
                "FXExposure": "Gas price in USD → tariff_gel correlates with xrate."
            }
        },
        "TariffContext": {
            "Definition": "Contextual regulatory and market factors explaining changes in electricity tariffs and their relationship to balancing prices.",
            "Rules": [
                "From May 2025, tariffs for Enguri HPP and Vardnili HPP Cascade increased due to a legislative amendment requiring these plants to cover the cost of electricity supplied to the occupied territory of Abkhazia by selling electricity to the rest of Georgia.",
                "From January 2024, regulated tariffs for all thermal power plants increased substantially because the procurement price of natural gas rose sharply after being fixed at a low level for several years.",
                "Renewable PPAs generally have higher fixed tariffs than average summer balancing prices; as renewable share grows and cheap hydro share declines, summer balancing prices converge toward average PPA prices.",
                "Balancing electricity is the residual of total generation minus volumes sold under bilateral contracts or on exchanges."
            ],
            "Interpretation": [
                "Tariff increases in 2024–2025 are primarily cost-driven, reflecting gas price rises, currency depreciation, and compensation mechanisms for unreimbursed energy.",
                "Seasonal price differences must be read in context of regulatory cost adjustments and evolving generation mix."
            ]
        },
        "AnalyticalImplications": [
            "Thermal tariffs rise with GEL depreciation and gas price increases.",
            "Hydro tariffs are stable across seasons.",
            "Guaranteed capacity ensures cost recovery even at low generation."
        ],
        "ExampleIndicators": [
            "Compare tariff_gel vs xrate to evaluate FX sensitivity.",
            "Compare p_gcap_gel vs tariff_gel to separate fixed vs variable cost effect."
        ]
    },

    "SeasonalityPatterns": {
        "SummerMonths": [4, 5, 6, 7],
        "WinterMonths": [1, 2, 3, 8, 9, 10, 11, 12],
        "Description": "Summer: hydro-dominant, low prices. Winter: thermal/import-dominant, high prices.",
        "AnalyticalUse": [
            "Compare prices and generation composition between seasons.",
            "Hydro share typically >60% in summer, <30% in winter."
        ]
    },

    "TariffDependencies": {
        "Enguri": "Reference hydro tariff – low, stable.",
        "GardabaniTPP": "New CCGT; tariff follows gas cost and xrate.",
        "OldTPPs": "Less efficient; higher tariffs, more volatile.",
        "UsageHint": "Compare tariff_gel with p_bal_gel to assess regulatory lag."
    },

    "SeasonalTrends": {
        "Definition": "Balancing electricity prices, generation, and demand exhibit structurally different behaviors across seasons due to shifts in supply composition and consumption patterns.",
        "Rule": "Always compute and compare seasonal averages and CAGRs for April–July (Summer) and August–March (Winter).",
        "Interpretation": [
            "Summer prices rise faster as cheap hydro shares decline and more output moves to contracts.",
            "Winter prices increase moderately due to higher gas costs and import reliance."
        ],
        "AnalyticalUse": [
            "Combine seasonal trend analysis with hydro, thermal, and import shares.",
            "Use SUM for quantities, AVG for prices when comparing seasons."
        ]
    },

    "BalancingMarketLogic": {
        "Definition": "The balancing market reflects short-term deviations between forecasted and actual generation or consumption.",
        "Rules": [
            "When hydro inflows are strong, surplus deregulated HPP output reduces balancing price and volume.",
            "Low-hydro months push balancing to thermal and imports, raising volatility and cost.",
            "Balancing prices reflect the residual mix, not just cost; cheap hydro depresses prices, gas/import raise them.",
            "Rising renewable PPA share lifts summer prices as it displaces cheap hydro from balancing volumes."
        ]
    },

    "TariffTransmissionMechanism": {
        "Definition": "Describes how regulatory tariff changes propagate through generation costs and market prices.",
        "KeyMechanisms": [
            "Thermal plant tariffs include fixed (capacity) and variable (gas-linked) components — gas price hikes pass through immediately.",
            "Enguri and Vardnili now recover full cost via higher tariff per sold MWh due to Abkhazia supply adjustment (2025).",
            "Renewable PPAs are USD-indexed and form a price floor for summer market prices.",
            "Thermal tariff increases (Mtkvari, Tbilisi, G-Power) transmit almost directly to winter balancing prices."
        ]
    },

    "ImportDependence": {
        "Definition": "Explains the strategic role of imports in price formation and adequacy.",
        "Rules": [
            "Georgia imports in winter, exports in summer; import exposure sets upper bound on domestic prices.",
            "Imports are USD-denominated and follow Turkish/Azeri prices, transmitting regional volatility.",
            "Higher import share + weaker GEL → higher balancing prices.",
            "Hydro shortfall or Enguri/Vardnili outages trigger import reliance and winter spikes."
        ]
    },

    "RenewableIntegration": {
        "Definition": "Captures how renewable PPAs affect market structure and balancing behavior.",
        "Rules": [
            "Renewable PPAs are fixed-price, USD-indexed, reducing residual balancing liquidity.",
            "Rising renewable share reduces hydro flexibility and increases balancing volatility.",
            "In summer, renewable PPAs can lift balancing price toward their own tariff level.",
            "In winter, thermal dominance limits renewable impact."
        ]
    },

    "DataEvidenceIntegration": {
        "Purpose": "Links conceptual rules in domain_knowledge with quantifiable evidence stored in Supabase materialized views.",
        "Guidance": [
            "Every analytical or causal statement should, when possible, be justified by trends or values from the corresponding materialized views.",
            "For tariff-related insights (e.g., Enguri/Vardnili increases, gas-cost effects), verify and illustrate with data from tariff_with_usd.",

            "Never include raw database column names (e.g., share_renewable_ppa, p_bal_usd, tariff_gel) in narrative text.",
            "regardless of the language of the response. Instead, use descriptive terms derived from domain_knowledge.",
            "or natural language equivalents (e.g., the share of renewable PPAs, the average balancing price in USD). If a suitable label is not found, infer a clear and human-readable name based on context before generating the answer.",

            
            "For balancing price behavior (summer vs winter, correlation with generation mix), use price_with_usd and trade_derived_entities.",
            "For demand or sectoral structure, reference energy_balance_long_mv.",
            "For import dependence or renewable share dynamics, use trade_derived_entities, focusing on share_import, share_deregulated_hydro, and share_renewable_ppa.",
            "If the user asks for an interpretation (not a raw figure), combine quantitative evidence (e.g., average, CAGR, or percentage change) with the relevant explanatory rule from domain_knowledge.",
            "If the user explicitly asks only for a numeric answer, provide the number directly without narrative justification.",
            "When comparing across currencies, units, or dimensions, ensure the interpretation explicitly reflects the measurement unit (e.g., GEL/MWh, USD/MWh, or share ratio)."
        ],
        "LLMHint": (
            "When generating an answer, use domain_knowledge for reasoning but cite patterns or magnitudes using actual computed results from the database views. "
            "Prioritize causal storytelling supported by numeric evidence rather than listing values. "
            "Balance narrative and precision: integrate data-driven observations with conceptual insights (e.g., rising prices explained by declining hydro share)."
        ),
        "ChartSelectionHint": (
            "When generating or explaining results, prioritize clarity over completeness. "
            "Select up to four key indicators that best answer the user's question — for example, price levels (in GEL/USD), main driver shares, and relevant exchange rate or tariff variables. "
            "Avoid including every numeric column in one chart unless the user explicitly asks for a full multi-series view. "
            "Prefer showing representative variables with clear contrast (e.g., GEL vs USD, hydro vs thermal, import share vs price) "
            "so the chart remains interpretable and visually balanced. "
            "When multiple variables differ by unit or scale, group them logically and use dual axes (left for price/tariff, right for share or index)."
        )
    }
}
