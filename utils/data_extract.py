from constants import technical_indicators, cash_flow_indicators, income_statement_indicators, END_DATE, START_DATE
from utils.helper import fetch_data_from_api
from collections import defaultdict
from datetime import datetime

def extract_technical_indicators(api_key, SYMBOL):
    data_extract = {}

    for key in technical_indicators:
        if technical_indicators[key]["enabled"]:
            data_extract[key] = fetch_data_from_api(
                technical_indicators[key]["function"],
                SYMBOL,
                technical_indicators[key]["interval"],
                technical_indicators[key].get("time_period", None),
                technical_indicators[key]["series_type"],
                api_key
            )[technical_indicators[key]["key_name"]]

            if key != "AROON":
                data_extract[key] = sorted([(k, float(data_extract[key][k][key])) for k in data_extract[key]])
            else:
                data_extract["AROON_UP"] = sorted([(k, float(data_extract[key][k]['Aroon Up'])) for k in data_extract[key]])
                data_extract["AROON_DOWN"] = sorted([(k, float(data_extract[key][k]['Aroon Down'])) for k in data_extract[key]])
                data_extract.pop("AROON")
    
    return data_extract

def extract_prices(api_key, SYMBOL):
    data = fetch_data_from_api("TIME_SERIES_DAILY", SYMBOL, api_key=api_key, outputsize="full")["Time Series (Daily)"]
    data_extracted = sorted([(key, float(data[key]['1. open'])) for key in data])
    return data_extracted

def extract_income_statement(api_key, SYMBOL):
    income_statement = fetch_data_from_api("INCOME_STATEMENT", SYMBOL, api_key=api_key)
    quarterly = income_statement["quarterlyReports"]
    feature_extract = defaultdict(lambda: [])
    for report in quarterly:
        for figure in income_statement_indicators:
            feature_extract[figure].append((report['fiscalDateEnding'], float(report[figure])))
    for key in feature_extract:
        feature_extract[key] = sorted(feature_extract[key])
    return feature_extract

def extract_cash_flow_statement(api_key, SYMBOL):
    cash_flow_statements = fetch_data_from_api("CASH_FLOW", SYMBOL, api_key=api_key)
    quarterly = cash_flow_statements["quarterlyReports"]
    feature_extract = defaultdict(lambda: [])
    for report in quarterly:
        for figure in cash_flow_indicators:
            feature_extract[figure].append((report['fiscalDateEnding'], float(report[figure])))
    for key in feature_extract:
        feature_extract[key] = sorted(feature_extract[key])
    return feature_extract

def extract_news(api_key, SYMBOL):
    end_time = ''.join(END_DATE.split("-")) + "T0000"
    start_time = ''.join(START_DATE.split("-")) + "T0000"
    article_data = fetch_data_from_api("NEWS_SENTIMENT", tickers=SYMBOL, api_key=api_key, limit=1000, time_from=start_time, time_to=end_time)
    article_feed = article_data["feed"]

    sentiment_by_date = [[datetime.strptime(n['time_published'], "%Y%m%dT%H%M%S"), n['ticker_sentiment']] for n in article_feed]
    for s in sentiment_by_date:
        for i in range(len(s[1])):
            if s[1][i]['ticker'] == SYMBOL:
                s[1] = float(s[1][i]['ticker_sentiment_score']) * float(s[1][i]['relevance_score'])
                break
    dates = [s[0] for s in sentiment_by_date]
    values = [100 * s[1] for s in sentiment_by_date]


    return article_data