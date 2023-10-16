technical_indicators = {
    "EMA": {
        "function": "EMA",
        "interval": "daily",
        "time_period": "5",
        "series_type": "open",
        "key_name": "Technical Analysis: EMA",
        "enabled": True
    },
    "RSI": {
        "function": "RSI",
        "interval": "daily",
        "time_period": "5",
        "series_type": "open",
        "key_name": "Technical Analysis: RSI",
        "enabled": True
    },
    "AROON": {
        "function": "AROON",
        "interval": "daily",
        "time_period": "5",
        "series_type": "open",
        "key_name": "Technical Analysis: AROON",
        "enabled": True
    },
    "MACD": {
        "function": "MACD",
        "interval": "daily",
        "series_type": "open",
        "key_name": "Technical Analysis: MACD",
        "enabled": True
    }
}

income_statement_indicators = ['netIncome', 'ebitda', 'grossProfit', 'operatingExpenses']

cash_flow_indicators = ['operatingCashflow', 'capitalExpenditures', 'cashflowFromInvestment', 'cashflowFromFinancing']