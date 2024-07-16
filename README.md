# Ticker-Predictor

This project uses python to create a long-short term recurrent neural network that projects the value of a stock over a given forecasting period. 

To use the model, 
1) First, get a AlphaVantage API key and enter the key into the 2nd line of the main function.
2) Then, replace the symbol and forecasting period in the main function.

This model uses the following mix of technical and fundamental indicators

Technical Indicators:
1. EMA
2. RSI
3. Last day closing price
4. AROON Up
5. AROON Down
6. MACD

Fundamental Indicators
1. Net Income, EBITDA, Gross Profit, Operating Income (Income Statement)
2. Operating Cash Flow, Capex, Cashflow from Investments, Cashflow from Financing (Cash Flow Statement)

Total (14 factor model)
