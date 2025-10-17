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

        "TariffContext": {
            "Definition": "Contextual regulatory and market factors explaining changes in electricity tariffs and their relationship to balancing prices.",
            "Rules": [
                "From May 2025, tariffs for Enguri HPP and Vardnili HPP Cascade increased due to a legislative amendment requiring these plants to cover the cost of electricity supplied to the occupied territory of Abkhazia by selling electricity to the rest of Georgia. Their tariff rose because they must recover total generation costs while being paid only for part of the produced electricity.",
                "From January 2024, regulated tariffs for all thermal power plants increased substantially because the procurement price of natural gas rose sharply after being fixed at a low level for several years.",
                "The average price of renewable PPAs is typically higher than the summer balancing market price. Balancing electricity—being the residual after bilateral or exchange trades—is often cheaper during summer due to a higher share of deregulated or regulated hydro generation.",
                "When the share of renewable PPA electricity increases and the shares of deregulated or regulated hydro decrease, the summer balancing price tends to converge toward the average PPA price.",
                "Balancing electricity represents the residual volume between total generation and electricity already sold under bilateral contracts or on exchanges."
            ],
            "Interpretation": [
                "Tariff increases in 2024–2025 are largely cost-driven, tied to gas import prices, legislative changes, and compensation for unreimbursed generation.",
                "Seasonal and structural price differences should be interpreted through these regulatory cost adjustments and generation-mix dynamics."
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


    "SeasonalTrends": {
        "Definition": "Balancing electricity prices, generation, and demand exhibit structurally different behaviors across seasons due to shifts in supply composition and consumption patterns.",
        "Rule": "Always compute and compare seasonal averages and CAGRs for April–July (Summer) and August–March (Winter) periods. These comparisons must always accompany the total yearly trend for balancing prices, generation, and demand. Seasonal analysis is not required for tariffs.",
        "DataBasis": "Seasonal composition and its evolution can be directly observed from the trade_derived_entities view, where the shares of deregulated_hydro and regulated_hpp reflect the availability of low-cost hydro generation in the balancing market, while import and thermal categories represent higher-cost sources.",
        "Interpretation": [
            "Summer prices tend to rise faster in recent years because the shares of cheap hydro sources (deregulated_hydro and regulated_hpp) in balancing electricity have declined. An increasing portion of hydro generation is sold through bilateral contracts instead of the balancing market.",
            "Winter prices also show an upward trend, though typically smaller, driven by higher gas costs for thermal generation and greater reliance on imports during low-hydro periods.",
            "Seasonal demand patterns amplify these effects: electricity consumption peaks in winter due to heating and again in summer due to cooling, affecting balancing volumes and price volatility."
        ],
        "AnalyticalUse": [
            "Combine seasonal trend analysis with hydro, thermal, and import share data from trade_derived_entities to explain structural shifts.",
            "When comparing Summer and Winter trends, always reference deregulated_hydro and regulated_hpp shares as key indicators of cheap hydro availability.",
            "When analyzing demand or generation, compare total quantities (SUM) by season, while for prices use averages (AVG)."
        ],
        "Insight": "Include both total yearly and separate seasonal trends (Summer vs Winter) in every balancing price, generation, or demand trend analysis, interpreting results through the evolution of hydro, thermal, and import shares and corresponding seasonal demand differences."
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
