import requests

def fetch_data_from_api(function, symbol=None, interval=None, time_period=None, series_type=None, api_key=None, topics=None, outputsize=None, limit=None, tickers=None, time_from=None, time_to=None):
    base_url = "https://www.alphavantage.co/query"
    
    # Initialize the dictionary with required parameters
    params = {
        "function": function,
        "symbol": symbol
    }
    
    # Add optional parameters if they are not None
    if interval is not None:
        params["interval"] = interval
    if time_period is not None:
        params["time_period"] = time_period
    if series_type is not None:
        params["series_type"] = series_type
    if api_key is not None:
        params["apikey"] = api_key
    if topics is not None:
        params["topics"] = topics
    if outputsize is not None:
        params["outputsize"] = outputsize
    if limit is not None:
        params["limit"] = limit
    if tickers is not None:
        params["tickers"] = tickers
    if time_from is not None:
        params["time_from"] = time_from
    if time_to is not None:
        params["time_to"] = time_to
        
    response = requests.get(base_url, params=params)
    return response.json()