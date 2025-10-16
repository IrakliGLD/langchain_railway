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
            "'ltd "mtkvari energy"', 'ltd "iec" (tbilresi)', and 'ltd "g power" (capital turbines)' are regulalted old TPP.",
            "'ltd "gardabni thermal power plant"' is regulalted new TPP.",
            "Energo-Pro hydro plants (Rioni, Lajanuri, Shaori, Gumati, Dzevruli) have similar cost structures and can be averaged together.",
            "Entity labels are provided for clearer chart legends and report outputs."
        ]
    },


    "price_with_usd": {
        "description": "Electricity market prices (including balancing electricity) converted to USD.",
        "dependencies": [
            "Balancing electricity price represents the weighted average price of all trades executed for balancing purposes.",
            "It depends on the volumes and prices of electricity sold as balancing energy from the 'trade' table.",
            "All electricity sold as balancing electricity is priced at the balancing price, which is the weighted average of electricity from different types of entities.",
            "Deregulated hydro and regulated HPP are typically the cheapest sources; regulated_new_TPP and thermal_PPA have similar mid-level prices; regulated_old_TPP and import are generally the most expensive.",
            "Renewable_PPA price is the main driver of balancing electricity price, particularly in summer, because it constitutes a major share of electricity sold as balancing electricity.",
            "Renewable_PPA is always sold as balancing electricity and cannot be sold under bilateral contracts or on exchanges during PPA months.",
            "Balancing electricity prices in summer are usually lower due to high hydro generation."
        ]
    },

    "trade": {
        "description": "Monthly trading data from market participants.",
        "dependencies": [
            "Includes transactions across exchange and balancing electricity segments.",
            "Trade volumes determine the weights used in calculating the balancing electricity price."
        ]
    },

    "CurrencyInfluence": {
        "GEL_USD_Effect": "Energy price analysis must consider the GEL/USD exchange rate. A change in the rate directly causes divergence between GEL-denominated and USD-denominated prices for the same product.",
        "USD_Denominated_Costs": [
            "Natural Gas for Thermal Generation (TGC) is primarily priced in USD. As GEL depreciates, the cost of thermal generation increases in GEL terms, pushing up wholesale and balancing prices.",
            "Electricity Imports are priced in USD. When GEL depreciates, import prices rise in GEL terms.",
            "Regulated tariffs (for distribution/supply) are often adjusted based on exchange rate changes to cover USD-denominated costs."
        ],
        "SeasonalityHint": "Price trends should be compared on a yearly average basis to neutralize the strong seasonal swings (high Hydro Generation in summer = low prices; high Thermal Generation in winter = high prices)."
    },

    "PriceDrivers": {
        "TargetColumns": ["p_bal_gel", "p_bal_usd"],
        "DriverColumns": {
            "p_dereg_gel": "Deregulated HPP Price (from price_with_usd)",
            "p_gardabani_tpp_tariff": "Regulated New TPP Tariff (from tariff_with_usd)",
            "p_grouped_old_tpp_tariffs": "Regulated Old TPP Group Tariffs (from tariff_with_usd)",
            "entity = 'deregulated_hydro'": "Deregulated HPP Share (from trade_derived_entities)",
            "entity = 'import'": "Import Share (from trade_derived_entities)",
            "entity = 'renewable_ppa'": "Renewable PPA Share (from trade_derived_entities)"
        }
    }
}
