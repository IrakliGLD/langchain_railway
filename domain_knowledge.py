DOMAIN_KNOWLEDGE = {
    "tariff_gen": {
        "description": "Regulated generation tariffs approved by the Georgian energy regulator (GNERC).",
        "dependencies": [
            "Tariffs are set using a cost-plus methodology defined by the regulator.",
            "They reflect approved investment costs, operational expenses, and the allowed return on assets.",
            "Tariffs of thermal generation units depend heavily on the USD exchange rate, because natural gas fuel costs are denominated in USD."
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
    
    # NEW STRUCTURE FOR CORRELATION DRIVERS
    "PriceDrivers": {
        "TargetColumns": ["p_bal_gel", "p_bal_usd"],
        "DriverColumns": {
            # Prices/Tariffs
            "p_dereg_gel": "Deregulated HPP Price (from price_with_usd)",
            "p_gardabani_tpp_tariff": "Regulated New TPP Tariff (from tariff_with_usd)",
            "p_grouped_old_tpp_tariffs": "Regulated Old TPP Group Tariffs (from tariff_with_usd)",
            # Shares/Volumes (These will be joined from trade_derived_entities)
            "share_deregulated_hydro": "Deregulated HPP Share (from trade_derived_entities)",
            "share_import": "Import Share (from trade_derived_entities)",
            "share_renewable_ppa": "Renewable PPA Share (from trade_derived_entities)",
            # Note: Regulated HPP and other hydro tariffs are grouped in 'tariff_with_usd'
        }
    }
}
