DOMAIN_KNOWLEDGE = {
    "tariff_gen": {
        "description": "Regulated generation tariffs approved by the energy regulator.",
        "dependencies": [
            "Tariffs are set based on cost-plus methodology defined by the regulator.",
            "Tariffs reflect approved investment, operational costs, and regulated return on assets.",
            "Tariffs may depend on exchange rates, fuel costs, and inflation (CPI)."
        ]
    },
    "price_with_usd": {
        "description": "Electricity market prices, including balancing electricity converted to USD.",
        "dependencies": [
            "Balancing electricity price represents the weighted average price of trades executed for balancing purposes.",
            "It depends on volumes and marginal prices from the 'trade' table.",
            "High volatility usually indicates imbalance between forecasted and actual generation."
        ]
    },
    "trade": {
        "description": "Hourly trading data from market participants.",
        "dependencies": [
            "Includes transactions across day-ahead, intraday, and balancing markets.",
            "Trade volumes drive price formation for both balancing and wholesale averages."
        ]
    },
    "monthly_cpi": {
        "description": "Monthly consumer price index for inflation adjustment.",
        "dependencies": [
            "Used to deflate nominal tariffs or prices into real terms for trend analysis."
        ]
    }
}
