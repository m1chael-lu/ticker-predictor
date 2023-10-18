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
WINDOW_SIZE = 3
FORECAST_LENGTH = 1
TRAINING_SPLIT = 0.90
SYMBOL = "KLIC"
EPOCHS = 55
BATCH_SIZE = 16
START_DATE = "2017-01-01"
END_DATE = "2023-01-01"
N = 30 # 30 last data points
K = 1 # Forecast Length
DEFAULT_TEXT_INPUTS = {
    "Training Split": "0.90",
    "Epochs": "55",
    "Stride": "3",
    "Window Length": "30",
    "Forecast Length": "1"
}
METRICS = ["MSE", "Accuracy", "Precision", "Recall", "F1 Score", "Future Price", "Current Price", "Dir Prediction"]