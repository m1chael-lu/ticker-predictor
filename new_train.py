import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from alpha_vantage.timeseries import TimeSeries
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras.models import Model
from keras import optimizers
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from collections import defaultdict
from datetime import datetime
from constants import technical_indicators, income_statement_indicators, cash_flow_indicators
from utils.helper import fetch_data_from_api
from keras.regularizers import l1_l2

WINDOW_SIZE = 50
FORECAST_LENGTH = 1
TRAINING_SPLIT = 0.80
SYMBOL = "XOM"
EPOCHS = 100
BATCH_SIZE = 32
START_DATE = "2019-01-01"
END_DATE = "2023-01-01"


N = 30 # 30 last data points
WINDOW_SIZE = 3 # 30 days
K = 1


api_key = os.environ['ALPHA_VANTAGE_API_KEY']


def extract_news(api_key):
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

    # # Create a DataFrame
    # df = pd.DataFrame({'Date': dates, 'Value': values})

    # weekly_avg = df.resample('W-Mon', on='Date').mean()

    # Extract year and month and create a new column 'YearMonth'
    # df['YearMonth'] = df['Date'].dt.to_period('M')

    # Group by 'YearMonth' and compute the mean for each group
    # monthly_avg = df.groupby('YearMonth').mean()

    # weekly_avg.index = weekly_avg.index.to_timestamp()

    # Plot the averaged values
    # plt.figure(figsize=(10, 6))
    # plt.plot(weekly_avg.index, weekly_avg['Value'], marker='o', color='blue')

    # # Labeling the plot
    # plt.title("Average Values by Month")
    # plt.xlabel("Month")
    # plt.ylabel("Average Value")
    # plt.grid(True)

    # plt.show()

    return article_data

def extract_technical_indicators(api_key):
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


def extract_prices(api_key):
    data = fetch_data_from_api("TIME_SERIES_DAILY", SYMBOL, api_key=api_key, outputsize="full")["Time Series (Daily)"]
    data_extracted = sorted([(key, float(data[key]['1. open'])) for key in data])
    return data_extracted

def extract_income_statement(api_key):
    income_statement = fetch_data_from_api("INCOME_STATEMENT", SYMBOL, api_key=api_key)
    quarterly = income_statement["quarterlyReports"]
    feature_extract = defaultdict(lambda: [])
    for report in quarterly:
        for figure in income_statement_indicators:
            feature_extract[figure].append((report['fiscalDateEnding'], float(report[figure])))
    for key in feature_extract:
        feature_extract[key] = sorted(feature_extract[key])
    return feature_extract

def extract_cash_flow_statement(api_key):
    cash_flow_statements = fetch_data_from_api("CASH_FLOW", SYMBOL, api_key=api_key)
    quarterly = cash_flow_statements["quarterlyReports"]
    feature_extract = defaultdict(lambda: [])
    for report in quarterly:
        for figure in cash_flow_indicators:
            feature_extract[figure].append((report['fiscalDateEnding'], float(report[figure])))
    for key in feature_extract:
        feature_extract[key] = sorted(feature_extract[key])
    return feature_extract

extracted_technical = extract_technical_indicators(api_key)
extracted_income = extract_income_statement(api_key)
ema, rsi, aroon_up, aroon_down = extracted_technical["EMA"], extracted_technical["RSI"], extracted_technical["AROON_UP"], extracted_technical["AROON_DOWN"]
stock_data = extract_prices(api_key)
extracted_news = extract_news(api_key)
extracted_cash_flow = extract_cash_flow_statement(api_key)
extracted_income.update(extracted_cash_flow)

def convert(date):
    year, month, day = date.split("-")
    year, month, day = int(year), int(month), int(day)
    return year * 365 + month * 30 + day

def convert_back(date):
    year = date // 365
    month = (date % 365) // 30
    day = (date % 365) % 30
    if month < 10:
        month = f"0{month}"
    if day < 10:
        day = f"0{day}"
    return f"{year}-{month}-{day}"

def generate_technical_data(extracted_data):
    generated_data = {}
    data_idx = {}
    for key in extracted_data:
        data_idx[key] = 0
        generated_data[key] = []
    
    price_idx, price_data, current_price_data = 0, [], []
    current_day = convert(START_DATE)
    end_day = convert(END_DATE)
    while current_day < end_day:
        # Technical Indicators
        for key in extracted_data:
            while convert(extracted_data[key][data_idx[key]][0]) < current_day:
                data_idx[key] += 1
            generated_data[key].append([value for _, value in extracted_data[key][data_idx[key]-N:data_idx[key]]])

        # Fundamental Indicators
        while convert(stock_data[price_idx][0]) < current_day + K:
            price_idx += 1
        
        current_price_data.append(stock_data[price_idx - K][1])
        price_data.append(stock_data[price_idx][1])

        current_day += WINDOW_SIZE
    return generated_data, current_price_data, price_data

def generate_single_factor_data(extracted_fundamental):
    fundamental_idx = defaultdict(lambda: 0)
    generated_fundamental = defaultdict(lambda: [])
    current_day = convert(START_DATE)
    end_day = convert(END_DATE)
    while current_day < end_day:
        # Income Statement Indicators
        for key in extracted_fundamental:
            while convert(extracted_fundamental[key][fundamental_idx[key]][0]) < current_day:
                fundamental_idx[key] += 1
            generated_fundamental[key].append(extracted_fundamental[key][fundamental_idx[key] - K][1])
        current_day += WINDOW_SIZE
    return generated_fundamental
        

def reshape_data(data):
    return data.reshape(data.shape[0], data.shape[1], 1)

# 1. Convert to NumPy arrays
# Convert each list into numpy arrays
generated_data, current_price_data, price_data = generate_technical_data(extracted_technical)
generated_fundamental = generate_single_factor_data(extracted_income)
for key in generated_data: generated_data[key] = reshape_data(np.array(generated_data[key]))
current_price_data_np, price_data_np = np.array(current_price_data), np.array(price_data)
for key in generated_fundamental: generated_fundamental[key] = np.array(generated_fundamental[key])


# Normalize data
scalers_x = {}

# Create a MinMaxScaler for each feature
for key in generated_data:
    scalers_x[key] = MinMaxScaler(feature_range=(0, 1))

# Stack them along the feature axis
technical_data_order = [key for key in generated_data]
for key in technical_data_order:
    generated_data[key][:, :, 0] = scalers_x[key].fit_transform(generated_data[key][:, :, 0]) 
technical_data_3d = np.concatenate([generated_data[key] for key in technical_data_order], axis=2)

scaler_y = MinMaxScaler(feature_range=(0, 1))

# Reshape current_price_data for concatenation
current_price_data_np = current_price_data_np[:, np.newaxis]

price_data_normalized = scaler_y.fit_transform(price_data_np.reshape(-1, 1))

fundamental_order = [key for key in generated_fundamental]
for key in fundamental_order:
    scalers_x[key] = MinMaxScaler(feature_range=(0, 1))

for key in fundamental_order:
    generated_fundamental[key] = scalers_x[key].fit_transform(generated_fundamental[key].reshape(-1, 1))

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(technical_data_3d, price_data_normalized, test_size=0.2, shuffle=False)

# Including single factor price into the generated fundamental
generated_fundamental['currentPrice'] = price_data_normalized

fundamental_order.append("currentPrice")

train_test_single_factor_splits = {}
for key in generated_fundamental:
    train_test_single_factor_splits[key] = train_test_split(generated_fundamental[key], test_size=0.2, shuffle=False)

X_single_train = np.concatenate([train_test_single_factor_splits[key][0] for key in fundamental_order if type(key) == str], axis=1)
X_single_test = np.concatenate([train_test_single_factor_splits[key][1] for key in fundamental_order if type(key) == str], axis=1)

# 2. Construct LSTM model with multiple inputs

# Technical indicators input branch
input_technical = Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm_tech = LSTM(25, return_sequences=True, kernel_initializer='he_normal')(input_technical)
lstm_tech = Dropout(0.2)(lstm_tech)
lstm_tech = LSTM(25, return_sequences=True)(lstm_tech)
lstm_tech = Dropout(0.2)(lstm_tech)
lstm_tech = LSTM(25)(lstm_tech)
lstm_tech_out = Dropout(0.2)(lstm_tech)

# Current price input branch
# Single Factor Input Branch
single_fact_input = Input(shape=(len(generated_fundamental.keys()),))
dense_single = Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(single_fact_input)

# Merge the outputs of the two branches
merged = concatenate([lstm_tech_out, dense_single])

# Add a final dense layer
output = Dense(1)(merged)

# Compile the model
model = Model(inputs=[input_technical, single_fact_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_train, X_single_train], y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)


# 6. Evaluate the model
y_pred = model.predict([X_test, X_single_test])
y_pred_original = scaler_y.inverse_transform(y_pred) # Convert back to original price scale
y_test_original = scaler_y.inverse_transform(y_test)

# Calculate MSE or any other error metric
mse = np.mean(np.square(y_pred_original - y_test_original))
print(f'Mean Squared Error on Test Data: {mse}')

y_train_pred = model.predict([X_train, X_single_train])
y_test_pred = model.predict([X_test, X_single_test])

# Directional accuracy
def directional_accuracy(y_true, y_pred):
    direction_true = np.sign(y_true[1:] - y_true[:-1])
    direction_pred = np.sign(y_pred[1:] - y_true[:-1])
    return np.mean(direction_true == direction_pred)

dir_acc = directional_accuracy(y_test_original, y_pred_original)
print(f'Directional Accuracy on Test Data: {dir_acc * 100:.2f}%')

# Combining data for plotting
actual_values = np.concatenate([scaler_y.inverse_transform(y_train), scaler_y.inverse_transform(y_test)])
predicted_values = np.concatenate([scaler_y.inverse_transform(y_train_pred), scaler_y.inverse_transform(y_test_pred)])

# Plotting the actual values
plt.figure(figsize=(14, 7))
plt.plot(actual_values, label="Actual Values", color='blue')

# Plotting the predicted values
plt.plot(predicted_values, label="Predicted Values", color='red', alpha=0.6)

# Marking the train-test split point on the plot
train_length = len(y_train)
plt.axvline(x=train_length, color='gray', linestyle='--', label='Train-Test Split')

# Labelling the plot
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

cash_flow = fetch_data_from_api("CASH_FLOW", SYMBOL, api_key=api_key)

def future_projection(model, stock_data, technical_data, income_data, cash_flow_data):
    single_factor = np.zeros((1, 1 + len(income_data.keys())))
    multifactor = np.zeros((1, N, len(technical_data.keys())))
    for i in range(len(technical_data_order)):
        key = technical_data_order[i]
        multifactor[0, :, i] = scalers_x[key].transform(np.array([value for _, value in technical_data[key][-N:]]).reshape((1, -1))).reshape(N)
    for i in range(len(fundamental_order)):
        key = fundamental_order[i]
        if key == "currentPrice":
            single_factor[0, i] = scaler_y.transform(np.array(stock_data[-1][1]).reshape((1, -1))).reshape(1)
        elif key in income_statement_indicators:
            single_factor[0, i] = scalers_x[key].transform(np.array(income_data[key][-1][1]).reshape((1, -1))).reshape(1)
        else:
            single_factor[0, i] = scalers_x[key].transform(np.array(cash_flow_data[key][-1][1]).reshape((1, -1))).reshape(1)
    return model.predict([multifactor, single_factor])

future_price = future_projection(model, stock_data, extracted_technical, extracted_income, extracted_cash_flow)



print(f"Predicted future price: {int(scaler_y.inverse_transform(future_price))}")
print(f"Current price: {stock_data[-1][1]}")
print(f"Up/Down Prediction: {'Up' if future_price > stock_data[-1][1] else 'Down'}")
print(f"Backtested Accuracy: {dir_acc * 100:.2f}%")
