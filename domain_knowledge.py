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


    "CfD_Contracts": {
        "description": "Contracts for Difference (CfD) introduced for new renewable power plant projects developed under Georgia's capacity auction scheme.",
        "key_facts": [
            "Georgia conducted several capacity auctions to support development of new hydro, solar, and wind power plants.",
            "All winning projects are renewable (hydro, solar, or wind), forming the first batch of CfD-based investments in the country.",
            "Under transitional market provisions, CfD projects are not allowed to sell electricity on the competitive exchange; instead, they are centrally dispatched by the system operator (GSE).",
            "The CfD ensures a fixed payment for generated electricity based on the agreed 'strike price' multiplied by the actual generated and accepted energy.",
            "If there were the self-dispatching regime and CfD owners get remunerated based on the sold electricity, the quantity risk would be minimized. So, Georgian Electricity Target Model, which envisages self-dispatching reduces the qauntity risk, but bringing the imbalance risk, whixh can be efficiently hedged.",
            "The 'reference market price' is used only for accounting and settlement between the offtaker and the market operator — it does not affect the generator’s final income.",
            "As a result, CfD owners are fully insulated from market price risk: they always receive the contracted CfD (strike) price for their accepted generation, regardless of the contemporaneous market price level."
        ],
        "market_implications": [
            "CfD plants face quantity risk, not price risk, because their remuneration depends solely on the amount of energy actually dispatched. If the system operator curtails their generation for system stability, the lost production is not compensated.",
            "Under a self-dispatch or exchange-participation model, producers could optimize generation to maximize revenue, but the CfD design transfers operational control to the central dispatcher.",
            "As more CfD plants enter operation, the central dispatch role of GSE becomes more complex, and overall balancing or curtailment volumes could increase depending on the variability of CfD output.",
            "Because CfD plants are centrally dispatched and remunerated outside the exchange, their energy bypasses market trading, which can reduce liquidity and price formation transparency.",
            "The effect of CfD generation on the **market price level** is uncertain: while additional renewable capacity increases total supply, most CfD projects are hydro and solar producing mainly in summer months. When the contracted CfD price is higher than contemporaneous market prices, their inclusion in system dispatch may indirectly raise the seasonal average price signal even though the generator itself remains price-neutral."
        ],
        "analytical_notes": [
            "Enai must treat CfD generators as **price-neutral** participants: their revenues are fixed at the strike price and are not influenced by short-term market price changes.",
            "Their main exposure is **quantity risk** — if GSE curtails output for system reasons, their remuneration falls proportionally.",
            "When analyzing average or balancing prices, Enai should note that CfD generation volumes are excluded from exchange transactions and remunerated separately, which can distort apparent price–volume relationships.",
            "Enai should describe CfD projects as a distinct category within generation — centrally dispatched, renewable, fully price-insulated, and subject to curtailment risk — with indirect but material influence on system balance and market price formation.",
            "All CfD-related quantities, price levels, and contract assumptions originate from the official GSE auction documentation and the Georgian electricity market transitional rules."
        ]
    },


    "BalancingMarketStructure": {
        "description": "Explanation of the current Georgian balancing mechanism and how it differs from the European-style balancing market design. This context is essential for Enai when interpreting balancing electricity prices or queries mentioning 'balancing market' or 'imbalance settlement'.",
        "current_design": [
            "Despite being formally called a 'balancing market', Georgia’s current system functions as an imbalance settlement mechanism rather than a real-time balancing market in the European sense.",
            "Balancing responsibility is not defined on an hourly basis — there is no concept of Balance Responsible Parties (BRPs) with continuous imbalance settlement.",
            "The current balancing period is one month, and the imbalance is calculated as the difference between the total electricity consumed or generated and the electricity sold or purchased during that same month."
        ],
        "price_determination": [
            "The balancing electricity price is calculated as a weighted average price of electricity sold as balancing energy during the month.",
            "This price formation principle is described in the 'price_with_usd' domain: it aggregates transactions across generation entities based on their quantities and individual tariffs or market values.",
            "Therefore, the current 'balancing price' represents a settlement value for deviations over a month, not an hourly marginal price determined by balancing actions or frequency restoration products."
        ],
        "comparison_with_eu_practice": [
            "In the European electricity market model, the balancing market includes trading of balancing products such as FCR (Frequency Containment Reserve), aFRR (Automatic Frequency Restoration Reserve), and mFRR (Manual Frequency Restoration Reserve).",
            "Those products are activated on a sub-hourly basis to maintain system frequency and resolve imbalances in real time, with separate procurement and activation prices.",
            "Georgia’s system does not yet have such hourly or product-based balancing. Instead, it performs monthly imbalance settlement after-the-fact, without separate reserve products or imbalance responsibility allocation."
        ],
        "future_direction": [
            "Full transition toward an EU-style balancing market is expected in future market reforms, with the introduction of BRPs, hourly metering, and separate balancing product procurement.",
            "Until that transition, Enai must interpret any reference to the 'balancing market' as meaning 'monthly imbalance settlement' rather than a real-time balancing product market."
        ],
        "analytical_notes": [
            "When Enai analyzes 'balancing price' trends, it must not treat them as real-time marginal prices — instead, they represent average monthly imbalance settlement values.",
            "Statements about balancing volumes or revenues refer to quantities of electricity settled as imbalances, not energy traded in real-time balancing product markets.",
            "In this context, Enai should link balancing price variations mainly to factors such as exchange rate movements, hydro/thermal generation composition, and tariff structures, as defined in the 'PriceDrivers' and 'CurrencyInfluence' domains."
        ]
    },


    "BalancingPriceFormation": {
        "Definition": "The balancing price is the weighted-average price of electricity sold on the balancing market.",
        "WeightingEntities": [
            "deregulated_hydro", "import", "regulated_hpp",
            "regulated_new_tpp", "regulated_old_tpp",
            "renewable_ppa", "thermal_ppa"
        ],
        "CalculationRule": "Weights are based on electricity sold as balancing energy by each entity. Total balancing quantity = sum of all listed entities from trade_derived_entities WHERE segment='balancing_electricity'.",
        "AnalyticalUse": [
            "Compute average balancing price weighted by balancing-market quantities.",
            "Compare composition changes (hydro vs thermal/import) seasonally."
        ],
        "Insight": "Prices rise in winter when thermal/import shares grow, and fall in summer when hydro dominates."
    },

    "BalancingPriceDrivers": {
        "CRITICAL": "This section contains the most important information for balancing price analysis.",
        "PrimaryDrivers": {
            "1_Composition": {
                "importance": "PRIMARY DRIVER - Most important for BOTH GEL and USD prices",
                "description": "Shares of different entity categories selling electricity on balancing segment",
                "categories": [
                    "renewable_ppa - Renewable PPA projects (hydro, solar, wind) under support schemes, USD-priced",
                    "deregulated_hydro - Deregulated hydropower plants, USD-priced",
                    "thermal_ppa - Thermal PPA projects, USD-priced",
                    "regulated_hpp - Regulated hydro power plants, GEL tariffs",
                    "regulated_old_tpp - Regulated old thermal power plants (Mtkvari, Tbilisi, G-POWER), GEL tariffs that directly reflect current xrate",
                    "regulated_new_tpp - Regulated new thermal power plant (Gardabani), GEL tariff that directly reflects current xrate",
                    "import - Direct electricity imports, USD-priced"
                ],
                "mechanism": [
                    "Balancing electricity price = weighted average of all categories selling on balancing segment",
                    "Higher share of cheap sources (regulated_hpp ~30-40 GEL/MWh, deregulated_hydro ~40-50 GEL/MWh) → lower price",
                    "Higher share of expensive sources (import, thermal_ppa, renewable_ppa) → higher price",
                    "Composition changes seasonally: summer=high renewable_ppa and hydro, winter=high thermal_ppa and import"
                ],
                "structural_trends": [
                    "IMPORTANT: Long-term declining trend in deregulated_hydro and regulated_hpp shares",
                    "IMPORTANT: Long-term increasing trend in renewable_ppa, import, and thermal_ppa shares",
                    "Main contributors to balancing electricity now: renewable_ppa (biggest in summer), import, thermal_ppa, regulated_old_tpp, regulated_new_tpp"
                ],
                "within_category_variation": [
                    "Individual plants within categories (especially renewable_ppa) have different USD prices",
                    "Depending on which individual plant contributes more, price can vary slightly within category",
                    "However, category-level analysis (renewable_ppa, deregulated_hydro) has the biggest impact on price"
                ],
                "data_source": "trade_derived_entities WHERE segment='balancing_electricity'",
                "analysis_requirement": [
                    "MANDATORY: For long-term trends or yearly analysis → check composition by season (summer vs winter)",
                    "MANDATORY: Cite actual share changes (e.g., 'renewable_ppa increased from 25.3% to 32.7%')",
                    "MANDATORY: Explain which categories are cheap vs expensive when explaining price impact",
                    "For month-to-month comparisons → compare share of each category between the two months"
                ]
            },
            "2_ExchangeRate": {
                "importance": "CRITICAL for GEL price, NO IMPACT on USD price",
                "variable": "xrate (GEL/USD)",
                "mechanism": [
                    "renewable_ppa, thermal_ppa, deregulated_hydro, import are all USD-priced",
                    "regulated_old_tpp and regulated_new_tpp tariffs are set in GEL BUT directly reflect current xrate",
                    "When GEL depreciates (xrate increases), GEL-denominated price rises",
                    "USD-denominated price is NOT affected by xrate (to see price excluding xrate effect, look at USD price)",
                    "All USD-priced categories and regulated TPP tariffs translate to higher GEL prices when xrate increases"
                ],
                "data_source": "price_with_usd view, column: xrate",
                "analysis_requirement": [
                    "For GEL price analysis: xrate is a major factor alongside composition",
                    "For USD price analysis: xrate has NO impact, only composition matters",
                    "regulated_old_tpp and regulated_new_tpp: GEL tariffs that adjust with xrate"
                ]
            }
        },
        "PriceHierarchy": {
            "description": "Price levels from cheapest to most expensive (approximate ranges for context)",
            "ranking": [
                "1. Regulated HPP (cheapest) - ~30-40 GEL/MWh",
                "2. Deregulated Hydro - varies, typically 40-50 GEL/MWh equivalent",
                "3. Regulated Thermal (old TPP, new TPP/Gardabani) - GEL tariffs that reflect current xrate and gas prices",
                "4. Renewable PPA - USD-priced, market-based (CONFIDENTIAL estimates for internal analysis)",
                "5. Thermal PPA - USD-priced, market-based (CONFIDENTIAL estimates for internal analysis)",
                "6. Import - USD-priced, market-based, typically expensive (CONFIDENTIAL estimates for internal analysis)"
            ],
            "CONFIDENTIALITY_RULE": "Specific PPA and import price estimates are for INTERNAL ANALYSIS ONLY. Never disclose these numbers to users. Say 'market-based' or 'varies' when discussing.",
            "SUPPORT_SCHEME_CLARIFICATION": "In Georgia, support schemes are PPA and CfD contracts ONLY. Regulated tariffs (regulated_hpp, regulated_old_tpp, regulated_new_tpp) are NOT support schemes - they are cost-plus regulated tariffs set by GNERC.",
            "data_sources": {
                "regulated_hpp_tariff": "tariff_with_usd view (tariff_gel, tariff_usd columns)",
                "regulated_thermal_tariff": "tariff_with_usd view (Gardabani, old TPPs) - GEL tariffs that reflect xrate",
                "deregulated_hydro_price": "price_with_usd view (p_dereg_gel, p_dereg_usd)",
                "renewable_ppa_price": "USD-priced, NOT IN DATABASE - market-based",
                "thermal_ppa_price": "USD-priced, NOT IN DATABASE - market-based",
                "import_price": "USD-priced, NOT IN DATABASE - market-based"
            }
        },
        "AnalysisGuidelines": {
            "for_correlation_analysis": [
                "Primary focus: composition (shares of each category in balancing electricity)",
                "Secondary focus: xrate (exchange rate) for GEL price analysis",
                "Calculate shares from trade_derived_entities WHERE segment='balancing_electricity'",
                "Cite correlation coefficients when available",
                "Note structural trends (declining deregulated_hydro/regulated_hpp, increasing renewable_ppa/import/thermal_ppa)"
            ],
            "for_price_explanation": [
                "STEP 1: Start by analyzing share_* columns for all categories",
                "STEP 2: Cite ACTUAL share changes with specific numbers (e.g., 'renewable_ppa increased from 25.3% to 32.7%')",
                "STEP 3: Explain price impact based on cheap vs expensive categories",
                "STEP 4: For GEL price, also analyze xrate change",
                "STEP 5: For long-term trends, separate summer vs winter composition analysis",
                "Always cite at least 2-3 main share changes when explaining price movement"
            ],
            "for_seasonal_analysis": [
                "Summer (April-July): Higher renewable_ppa and deregulated_hydro shares → lower prices",
                "Winter (Aug-March): Higher thermal_ppa and import shares → higher prices",
                "For multi-year trends: Calculate average shares for summer vs winter separately",
                "Note: Seasonal composition difference is the main driver of seasonal price difference"
            ],
            "disclosure_rules": [
                "DO disclose: regulated tariffs (~30-40 GEL/MWh for HPP), correlation coefficients, structural trends",
                "DO disclose: That regulated TPP tariffs (while in GEL) reflect current xrate",
                "DO NOT disclose: Specific renewable PPA prices (say 'market-based')",
                "DO NOT disclose: Specific thermal PPA prices (say 'market-based')",
                "DO NOT disclose: Specific import prices (say 'market-based')",
                "DO clarify: Support schemes = PPA + CfD only (NOT regulated tariffs)"
            ]
        }
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


    "TransmissionNetworkDevelopment": {
        "description": "Long-term transmission system priorities and constraints identified in the 2024–2034 Ten-Year Network Development Plan (TYNDP) prepared by Georgian State Electrosystem (GSE). All data, capacity values, and investment figures used by the model must explicitly refer to this document as the source: 'GSE Ten-Year Network Development Plan 2024–2034 (TYNDP)'.",
        "main_objectives": [
            "Ensure security of supply and system reliability through meeting N-1, G-1, and N-G-1 criteria.",
            "Address west–east transmission imbalance caused by concentration of hydro generation in western Georgia and main consumption in the east (Tbilisi–Rustavi).",
            "Eliminate critical bottlenecks along the Enguri–Zestaponi–Imereti 500/220 kV corridor and strengthen internal transfer capacity.",
            "Modernize substations and increase transformer capacities to maintain reliability under growing urban demand (especially Tbilisi and Batumi)."
        ],
        "renewable_integration": [
            "Up to 750 MW of new wind and 500 MW of new solar capacity can be integrated by 2028 under existing balancing and regulating resources.",
            "Integration of renewables is contingent on new flexible capacity (CCGT) and reservoir HPPs to provide balancing and system inertia.",
            "The west–east imbalance intensifies during high-hydro summer periods, reinforcing the need for transmission upgrades and storage solutions."
        ],
        "investment_program": [
            "16 priority projects were identified, totaling approximately 1,400 km of new or rehabilitated lines and 5,000 MVA of transformer capacity.",
            "68% of these projects are expected to be commissioned during 2024–2026, 17% in 2027–2028, and the remainder by 2034.",
            "Total expected investment is around 1 billion EUR, primarily aimed at improving reliability, enabling integration of 3.5–4 GW of new hydro capacity, and expanding cross-border transit potential."
        ],
        "cross_border_projects": [
            "Georgia–Romania Black Sea Submarine Cable (HVDC) — will connect Georgia to the Continental European grid via Romania, enabling direct electricity trade with EU markets.",
            "Georgia–Russia–Azerbaijan Power System Connection — aims to enable synchronous or coordinated operation and controlled power flows within the regional IPS/UPS network.",
            "Additional interconnections with Turkey and Armenia are planned to strengthen regional exchange and improve system flexibility."
        ],
        "digital_and_control_infrastructure": [
            "Development of SCADA, WAMS, and ACS systems integrated with IT and cybersecurity frameworks is part of the ten-year plan to enhance real-time system observability and control.",
            "Advanced automation and information platforms are required to manage a more decentralized, variable, and renewable-heavy power system."
        ],
        "analytical_notes": [
            "The leitmotif of Georgian transmission development over 2024–2034 is 'security of supply' — maintaining system stability amid rapid generation diversification and demand growth.",
            "Transmission bottlenecks directly affect the marginal cost of balancing electricity; congestion between western hydro generation and eastern demand zones can elevate balancing prices even in high-hydro months.",
            "Enai should treat the transmission plan as a structural constraint in price and dispatch analysis: west–east congestion and delayed network reinforcements can explain short-term price spreads or curtailment patterns despite sufficient installed capacity.",
            "Whenever Enai references line capacities, regional transfer limits, or project commissioning dates, it must cite: 'Source: GSE Ten-Year Network Development Plan 2024–2034 (TYNDP)'."
        ]
    },

    "GenerationAdequacyAndForecast": {
        "description": "Long-term forecast of generation capacity, demand, and adequacy from the Georgian TSO’s 2024–2034 Ten-Year Network Development Plan (TYNDP). All quantitative figures (installed capacities, demand forecasts, or adequacy margins) must be attributed to this document as the official source.",
        "current_state": [
            "As of 2023, total installed capacity in Georgia was about 4,621 MW, of which 73% was hydropower (51% regulating, 22% seasonal), 23% thermal, and 0.5% wind.",
            "The generation–demand pattern is highly seasonal: in summer, hydro generation exceeds demand allowing exports; in winter, hydro output declines and thermal generation covers up to 28% of total supply."
        ],
        "forecast_to_2034": [
            "Total installed capacity is expected to reach around 10,336 MW by 2034, more than doubling the current level.",
            "Of this, 4,314 MW will be regulating HPPs, 2,908 MW seasonal HPPs, 1,291 MW wind, 232 MW solar, 110 MW gas turbines, and 1,480 MW combined-cycle and existing Gardabani TPP units.",
            "Hydropower will remain dominant, with about 70% of total capacity, while wind and solar will together provide roughly 15%.",
            "This capacity mix aims to reduce import dependence in winter and fossil-fuel exposure, but will increase seasonal and spatial variability of generation."
        ],
        "demand_and_balance_scenarios": [
            "Three demand growth scenarios are modeled — L1 (1% annual), L2 (3% base case), and L3 (5%) — paired with generation build-out cases G1, G2, and G3 representing different commissioning delays.",
            "Base-case assumption (L2G3) foresees 3% annual consumption growth and on-time integration of all planned generation projects.",
            "The plan anticipates that under all scenarios, Georgia will maintain energy adequacy through 2034, but seasonal surpluses in summer and shortages in winter will persist."
        ],
        "generation_adequacy_analysis": [
            "Adequacy simulations using the PLEXOS model confirm sufficient capacity under all scenarios, but highlight the importance of adding regulating HPPs and CCGTs for flexibility.",
            "Wind and solar additions contribute to overall adequacy, yet their variability requires reserve margins and flexible hydropower for balancing.",
            "Scenario G2 (with reservoir hydro development) provides higher flexibility and resilience compared to G1 (mostly run-of-river renewables)."
        ],
        "analytical_notes": [
            "Enai should interpret forecasts and adequacy results as system-level constraints influencing price and balancing behavior.",
            "Summer export surpluses will continue to dominate the balance, potentially lowering domestic balancing prices during high-hydro years, while winter scarcity will sustain higher balancing and import prices.",
            "When explaining trends, Enai must explicitly state that figures and forecasts originate from the GSE TYNDP 2024–2034, and link seasonal price patterns to the projected evolution of regulating versus seasonal hydro and the limited contribution of solar and wind during winter peaks."
        ]
    },


    "MarketParticipantsAndDataSources": {
        "description": "Key institutions in Georgia’s electricity sector, their primary functions, and mapping between system data views and source organizations.",
        "participants": {
            "GNERC (Georgian National Energy and Water Supply Regulatory Commission)": {
                "role": "Independent energy regulator, tariff authority, energy market monitoring and licensing body.",
                "functions": [
                    "In electricity sector pproves electricity generation, transmission, and distribution tariffs, including tariff methodologies.",
                    "Issues, modifies, and revokes licenses for generation, transmission, distribution, martket operator. Authorized electricity activities not subject to license, like suppy, trade, small generation.",
                    "Approves and enforces the Grid Code, network connection rules, and accounting standards for market participants.",
                    "Oversees cost audits, tariff reviews, guaranteed capacity payments, and consumer protection measures."
                ],
                "notes": [
                    "GNERC is the primary source for the 'tariff_with_usd', 'price_with_usd' views.",
                    "It also collects and validates generation technology reports used in 'tech_quantity_view' and similar datasets."
                ]
            },
            "ESCO (Electricity System Commercial Operator)": {
                "role": "Electricity System Commercial Operator. Responsible for buying and selling balancing electricity.",
                "functions": [
                    "Administers the balancing and guaranteed capacity settlement processes.",
                    "Registers wholesale market participants, manages direct contracts.",
                    "Handles import/export settlements and acts as counterparty for CfD (Contract for Difference) and guaranteed capacity contracts."
                ],
                "notes": [
                    "ESCO provides data for 'trade_derived_entities', and other balancing-related views.",
                    "Although named a 'balancing market', the current ESCO mechanism operates as a monthly imbalance settlement system rather than an hourly balancing product market."
                ]
            },
            "GSE (Georgian State Electrosystem)": {
                "role": "Transmission System Operator (TSO), system dispatcher, and transmission network owner.",
                "functions": [
                    "Owns and operates Georgia’s transmission infrastructure.",
                    "Performs real-time system dispatch, grid stability control.",
                    "Manages cross-border interconnections with neighboring systems (Turkey, Azerbaijan, Armenia, Russia).",
                    "Plans transmission development and publishes the Ten-Year Network Development Plan (TYNDP)."
                ],
                "notes": [
                    "GSE is responsible for the operation and reliability of the national grid but does not generate electricity.",
                    "GSE is licensed operator of a balanicng market, however, real balancing market and hourly imbalance responsibility was set to launch on july 2027."
                ]
            },
            "GENEX (Georgian Energy Exchange)": {
                "role": "Electricity Exchange Operator for electricity day-ahead and intraday markets.",
                "functions": [
                    "Operates day-ahead, intraday.",
                    "Publishes market prices, traded volumes, and clearing results.",
                    "Handles financial settlement of trades executed on the exchange."
                ],
                "notes": [
                    "GENEX was established jointly by GSE and ESCO in 2019. Current shareholders are GSE, ESCO, GGTC and GOGC to implement the competitive wholesale electricity market framework.",
                ]
            },
            "GEOSTAT (National Statistics Office of Georgia)": {
                "role": "Official statistical agency providing macroeconomic and energy data.",
                "functions": [
                    "Publishes national energy balances, sectoral demand indicators, and inflation indices.",
                    "Maintains the CPI (Consumer Price Index) series, including the 'electricity, gas, and other fuels' category used in energy affordability analysis.",
                    "Provides long-term energy balance datasets in cooperation with GSE and GNERC."
                ],
                "notes": [
                    "GEOSTAT is the data source for 'monthly_cpi_mv' and 'energy_balance_long_mv'.",
                    "Enai should treat GEOSTAT datasets as official statistical references for national-level demand and macroeconomic trends."
                ]
            }
        },
        "data_source_mapping": {
            "price_with_usd, tariff_with_usd": "GNERC – Georgian National Energy and Water Supply Regulatory Commission",
            "monthly_cpi_mv": "GEOSTAT – National Statistics Office of Georgia",
            "energy_balance_long_mv": "GEOSTAT – National Statistics Office of Georgia",
            "tech_quantity_view (or tech_quantity_pivot)": "GNERC – generation reports from licensed producers",
            "trade_derived_entities, other operational views": "ESCO / GENEX – balancing and market settlement data"
        },
        "analytical_notes": [
            "When Enai references tariff data, it must cite GNERC as the source of tariff methodologies and approved rates.",
            "For macro energy balances and CPI statistics, GEOSTAT is the authoritative source.",
            "For generation by technology (hydro, thermal, wind, solar), Enai should attribute data to GNERC, based on licensee reporting.",
            "All balancing and market operation statistics are based on ESCO and GENEX datasets unless otherwise noted.",
            "When explaining discrepancies or missing data, Enai should note possible timing lags between GNERC, ESCO, and GEOSTAT publications."
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

    "BalancingPriceDecomposition": {
        "Definition": "Methodology for decomposing the weighted-average balancing price into entity-level contributions to explain price dynamics.",
        "Purpose": "Enable analysis of which entities drive balancing price changes by combining quantity shares with entity-specific prices.",
        "AvailableFunctions": {
            "compute_entity_price_contributions": {
                "description": "Calculates monthly entity contributions to balancing price using available reference prices",
                "output_columns": [
                    "balancing_price_gel: actual weighted average balancing price",
                    "share_[entity]: quantity share of each entity in balancing electricity",
                    "price_[entity]: reference price for entity (from tariff_with_usd or price_with_usd)",
                    "contribution_[entity]: estimated contribution = share × reference_price",
                    "total_known_contributions: sum of all calculable contributions",
                    "residual_contribution_ppa_import: unexplained portion (entities without price data)"
                ],
                "data_availability": {
                    "available": [
                        "Regulated HPP: average tariff from main hydro plants (Enguri, Vardnili, Energo-Pro)",
                        "Deregulated hydro: p_dereg_gel from price_with_usd",
                        "Regulated new TPP: Gardabani tariff_gel from tariff_with_usd",
                        "Regulated old TPPs: average of Mtkvari, Tbilisi, G-Power tariffs"
                    ],
                    "unavailable": [
                        "Renewable PPA: prices are confidential, not in database",
                        "Thermal PPA: prices are confidential, not in database",
                        "Import: prices vary by transaction, not in database"
                    ]
                },
                "usage": "Use to identify which entities contributed most to price increases/decreases month-over-month"
            },
            "compute_share_changes": {
                "description": "Calculates month-over-month changes in entity shares to track composition shifts",
                "output_columns": [
                    "share_[entity]: current month share",
                    "prev_share_[entity]: previous month share",
                    "change_share_[entity]: absolute change in percentage points",
                    "price_change_gel: corresponding change in balancing price"
                ],
                "usage": "Use to correlate price changes with composition changes (e.g., if import share increased by 15pp and price rose by 20 GEL/MWh)"
            }
        },
        "AnalyticalWorkflow": [
            "1. Use compute_share_changes to identify which entity shares changed significantly month-over-month",
            "2. Use compute_entity_price_contributions to estimate the price impact of those share changes",
            "3. Consider xrate changes for GEL-denominated price analysis (affects thermal, import, PPA costs)",
            "4. Link composition shifts to seasonal patterns (summer=hydro dominant, winter=thermal/import dominant)",
            "5. For entities without price data (PPAs, imports), infer direction from residual_contribution_ppa_import"
        ],
        "Limitations": [
            "Reference prices (tariffs) may differ from actual balancing transaction prices",
            "PPA and import prices not available; residual contribution provides proxy only",
            "Does not capture intra-month price volatility, only monthly averages",
            "Assumes linear price-quantity relationships, which may not hold for marginal pricing"
        ],
        "InterpretationGuidelines": [
            "A positive contribution increase indicates that entity contributed more to raising the price",
            "Compare contribution changes to price changes to assess relative importance",
            "Large residual_contribution_ppa_import suggests PPAs/imports drove price, but exact decomposition unknown",
            "Always validate decomposition insights against seasonal patterns and tariff changes"
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
    },

    "TableSelectionGuidance": {
        "Purpose": "Provides clear rules for choosing the appropriate database view based on query intent.",
        "Rules": [
            "When the query needs ONLY technical generation/consumption data (quantities by technology type):",
            "  → Use tech_quantity_view",
            "  → Contains: quantity by type_tech (hydro, thermal, wind, solar, import, demand-side types)",
            "  → Use for: demand trends, supply trends, generation mix by technology",
            "  → Examples: 'Show me demand trends', 'Hydro generation over time', 'Import quantities'",
            "",
            "When the query needs trade information, market prices, or entity-level analysis:",
            "  → Use trade_derived_entities",
            "  → Contains: traded quantities by entity and segment (balancing_electricity, bilateral contracts, exchange)",
            "  → Contains: entity shares (share_import, share_renewable_ppa, share_deregulated_hydro, etc.)",
            "  → Use for: balancing price analysis, composition changes, entity market behavior",
            "  → Examples: 'Explain balancing price variations', 'Entity shares in balancing market', 'Trade volumes by segment'",
            "",
            "CRITICAL DISTINCTION:",
            "  tech_quantity_view = Technical data (generation/demand by technology)",
            "  trade_derived_entities = Market/trade data (prices, segments, entities, shares)",
            "",
            "Default preference:",
            "  - Simple quantity queries → tech_quantity_view",
            "  - Price explanation queries → trade_derived_entities",
            "  - If uncertain, prefer tech_quantity_view for cleaner technical analysis"
        ]
    },

    "EnergySecurityAnalysis": {
        "Purpose": "Provides correct framework for analyzing energy security and import dependence.",
        "CriticalFact": "Thermal generation uses imported natural gas and cannot be considered fully domestic/local generation.",
        "CorrectClassification": {
            "LocalGeneration": [
                "Hydro (all types: regulated HPP, deregulated hydro, reservoir, run-of-river)",
                "Wind (renewable, no fuel imports)",
                "Solar (renewable, no fuel imports)",
                "Note: These have NO import dependence for fuel/energy input"
            ],
            "ImportDependentGeneration": [
                "Thermal (uses imported natural gas for power generation)",
                "Direct electricity import",
                "Note: Both depend on cross-border energy supply"
            ]
        },
        "DataSource": {
            "table": "tech_quantity_view",
            "description": "Use tech_quantity_view with understanding that thermal + import = import-dependent",
            "usage": "For energy security: sum thermal + import as import-dependent, sum hydro + wind + solar as local"
        },
        "AnalyticalImplications": [
            "When analyzing energy security, thermal is NOT a substitute for imports — it IS import-dependent",
            "The real choice for Georgia is: import electricity OR import gas to generate electricity",
            "True energy independence comes from hydro, wind, and solar expansion",
            "Winter import dependence = direct electricity imports + gas imports for thermal generation",
            "Summer energy surplus is real because it's based on local hydro without fuel imports"
        ],
        "ExampleStatements": [
            "✅ CORRECT: 'Georgia's energy security depends on local renewables (hydro, wind, solar). Thermal generation, while domestic, relies on imported gas and does not reduce import dependence.'",
            "✅ CORRECT: 'In winter, Georgia is import-dependent: ~30% direct electricity import + thermal generation using imported gas.'",
            "❌ WRONG: 'Thermal generation is local production that reduces import dependence.'",
            "❌ WRONG: 'Georgia can achieve energy independence by increasing thermal capacity.'"
        ]
    },

    "PriceComparisonRules": {
        "Purpose": "Mandatory rules for comparing electricity prices across time periods.",
        "CriticalRule": "ALWAYS mention summer and winter averages separately when comparing prices — never use annual averages only.",
        "Reasoning": [
            "Summer and winter prices are structurally different due to generation mix",
            "Summer: hydro-dominant, lower prices (~40-60 GEL/MWh)",
            "Winter: thermal/import-dominant, higher prices (~80-120 GEL/MWh)",
            "Annual averages obscure these critical seasonal differences"
        ],
        "TrendDivergence": {
            "SummerTrend": [
                "Rising share of renewable PPA and CfD scheme generation",
                "Reason: In absence of liquid markets, investors require government support schemes",
                "New renewable development depends almost entirely on PPA/CfD support",
                "As renewable PPA/CfD share increases in balancing electricity, summer prices converge toward average support scheme price",
                "This pushes summer prices UP over time despite high hydro availability"
            ],
            "WinterTrend": [
                "Prices follow gas prices and exchange rate (thermal/import dominant)",
                "Less affected by renewable PPA/CfD trend (thermal dominates in winter)",
                "More volatile due to import price variations and gas market dynamics"
            ]
        },
        "MandatoryFormat": [
            "When comparing prices across years:",
            "  1. State summer average for period A",
            "  2. State winter average for period A",
            "  3. State summer average for period B",
            "  4. State winter average for period B",
            "  5. Explain the different drivers for each season's trend"
        ],
        "ExampleStatements": [
            "✅ CORRECT: 'Balancing prices increased from 2020 to 2024. In summer, average prices rose from 45 GEL/MWh to 62 GEL/MWh due to growing renewable PPA share displacing cheap hydro from balancing. In winter, prices increased from 95 GEL/MWh to 118 GEL/MWh primarily due to gas price increases and GEL depreciation.'",
            "❌ WRONG: 'Balancing prices increased from 70 GEL/MWh in 2020 to 90 GEL/MWh in 2024.' (No seasonal breakdown!)"
        ]
    }
}
