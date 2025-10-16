DOMAIN_KNOWLEDGE = {
    "tariff_entities": {
        "description": "Specific generator entities present in the tariff_with_usd view, used for tariff-based correlation and chart labeling.",
        "hydro": {
            "engurhesi": {
                "entities": ['ltd "engurhesi"1'],
                "labels": {"ltd \"engurhesi\"1": "Enguri HPP"}
            },
            "other_hydro": {
                "entities": [
                    'jsc "energo-pro georgia genration" (dzevrulhesi)',
                    'jsc "energo-pro georgia genration" (gumathesi)',
                    'jsc "energo-pro georgia genration" (shaorhesi)',
                    'jsc "energo-pro georgia genration" (rionhesi)',
                    'jsc "energo-pro georgia genration" (lajanurhesi)',
                    'jsc "georgian water & power" (zhinvalhesi)',
                    'ltd "vardnili hpp cascade"',
                    'ltd "vartsikhe-2005"',
                    'ltd "khrami_1"',
                    'ltd "khrami_2"'
                ],
                "labels": {
                    'jsc "energo-pro georgia genration" (dzevrulhesi)': "Dzevruli HPP",
                    'jsc "energo-pro georgia genration" (gumathesi)': "Gumati HPP",
                    'jsc "energo-pro georgia genration" (shaorhesi)': "Shaori HPP",
                    'jsc "energo-pro georgia genration" (rionhesi)': "Rioni HPP",
                    'jsc "energo-pro georgia genration" (lajanurhesi)': "Lajanuri HPP",
                    'jsc "georgian water & power" (zhinvalhesi)': "Zhinvali HPP",
                    'ltd "vardnili hpp cascade"': "Vardnili HPP Cascade",
                    'ltd "vartsikhe-2005"': "Vartsikhe HPP",
                    'ltd "khrami_1"': "Khrami I HPP",
                    'ltd "khrami_2"': "Khrami II HPP"
                }
            }
        },
        "thermal": {
            "entities": [
                'ltd "gardabni thermal power plant"',
                'ltd "mtkvari energy"',
                'ltd "iec" (tbilresi)',
                'ltd "g power" (capital turbines)',
            ],
            "labels": {
                'ltd "gardabni thermal power plant"': "Gardabani TPP",
                'ltd "mtkvari energy"': "Mtkvari Energy",
                'ltd "iec" (tbilresi)': "Tbilisi TPP",
                'ltd "g power" (capital turbines)': "G-POWER"
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
            "Includes transactions across exchange and balancing segments.",
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
        "SummerMonths": [4,5,6,7],
        "WinterMonths": [1,2,3,8,9,10,11,12],
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

    "InflationLinks": {
        "Relation": "Electricity tariffs partially track CPI for 'electricity, gas and other fuels'.",
        "UseCase": "Compare CPI vs tariff_gel or p_bal_gel.",
        "Insight": "Divergence indicates lagged cost pass-through."
    },

    "EnergySectorTrends": {
        "Variables": ["sector", "energy_source", "volume_tj"],
        "Questions": [
            "Which sector’s energy use grew fastest?",
            "Did households or industry drive demand growth?"
        ],
        "Rules": "Use energy_balance_long_mv grouped by sector and energy_source; compute shares."
    },

    "OwnershipPatterns": {
        "OwnershipTypes": ["State", "Private", "Mixed"],
        "EntityExamples": {
            "State": ["Engurhesi", "GSE", "GWP"],
            "Private": ["Energo-Pro Georgia Generation", "Gardabani TPP"]
        },
        "AnalyticalUse": "Group tariffs or generation by ownership type to assess performance."
    },

    "TradeSegments": {
        "SegmentTypes": ["balancing", "exchange", "bilateral"],
        "DerivedShares": "Compute shares per segment and entity using quantity / SUM(quantity) by date.",
        "AnalyticalUse": "Identify whether balancing segment grows during low hydro periods."
    },

    "DerivedDimensions": {
        "season": {
            "definition": "A derived analytical grouping dividing the year into two structural periods.",
            "rules": {
                "Summer": [4,5,6,7],
                "Winter": [1,2,3,8,9,10,11,12]
            },
            "use_cases": [
                "When analyzing prices, tariffs, or generation volumes, compute seasonal averages or totals.",
                "For prices: take AVG() by season to compare hydro vs thermal.",
                "For quantities: take SUM() by season to measure total energy generated or consumed."
            ],
            "sql_hint": """
            SELECT
              CASE WHEN EXTRACT(MONTH FROM date) IN (4,5,6,7) THEN 'Summer' ELSE 'Winter' END AS season,
              AVG(p_bal_gel) AS avg_balancing_price_gel
            FROM price_with_usd
            GROUP BY season;
            """,
            "comment": "Season is not a database column but a derived field computed from EXTRACT(MONTH FROM date)."
        }
    }
}
