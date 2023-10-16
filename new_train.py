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
import bisect

WINDOW_SIZE = 50
FORECAST_LENGTH = 1
TRAINING_SPLIT = 0.80
SYMBOL = "AMZN"
EPOCHS = 100
BATCH_SIZE = 32

def fetch_data_from_api(function, symbol, interval=None, time_period=None, series_type=None, api_key=None, topics=None, outputsize=None):
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
        
    response = requests.get(base_url, params=params)
    return response.json()

N = 30 # 10 last data points
WINDOW_SIZE = 30 # 45 days
K = 1


api_key = os.environ['ALPHA_VANTAGE_API_KEY']

technical_indicators = {
    "EMA": {
        "function": "EMA",
        "interval": "weekly",
        "time_period": "10",
        "series_type": "open",
        "key_name": "Technical Analysis: EMA",
        "enabled": True
    },
    "RSI": {
        "function": "RSI",
        "interval": "weekly",
        "time_period": "10",
        "series_type": "open",
        "key_name": "Technical Analysis: RSI",
        "enabled": True
    },
    "AROON": {
        "function": "AROON",
        "interval": "weekly",
        "time_period": "10",
        "series_type": "open",
        "key_name": "Technical Analysis: AROON",
        "enabled": True
    },
}

def extract_technical_indicators(api_key):
    data_extract = {}

    for key in technical_indicators:
        if technical_indicators[key]["enabled"]:
            data_extract[key] = fetch_data_from_api(
                technical_indicators[key]["function"],
                SYMBOL,
                technical_indicators[key]["interval"],
                technical_indicators[key]["time_period"],
                technical_indicators[key]["series_type"],
                api_key
            )[technical_indicators[key]["key_name"]]

            if key != "AROON":
                data_extract[key] = sorted([(k, float(data_extract[key][k][key])) for k in data_extract[key]])
            else:
                data_extract["AROON_UP"] = sorted([(k, float(data_extract[key][k]['Aroon Up'])) for k in data_extract[key]])
                data_extract["AROON_DOWN"] = sorted([(k, float(data_extract[key][k]['Aroon Down'])) for k in data_extract[key]])
    
    return data_extract


def extract_prices(api_key):
    data = fetch_data_from_api("TIME_SERIES_DAILY", SYMBOL, api_key=api_key, outputsize="full")["Time Series (Daily)"]
    data_extracted = sorted([(key, float(data[key]['1. open'])) for key in data])
    return data_extracted

extracted = extract_technical_indicators(api_key)
ema, rsi, aroon_up, aroon_down = extracted["EMA"], extracted["RSI"], extracted["AROON_UP"], extracted["AROON_DOWN"]
stock_data = extract_prices(api_key)

START_DATE = "2010-01-01"
END_DATE = "2022-01-01"

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

def generate_technical_data():
    ema_idx, rsi_idx, aroon_up_idx, aroon_down_idx, price_idx = 0, 0, 0, 0, 0
    ema_data, rsi_data, aroon_up_data, aroon_down_data, price_data, current_price_data = [], [], [], [], [], []

    current_day = convert(START_DATE)
    end_day = convert(END_DATE)
    while current_day < end_day:
        while convert(ema[ema_idx][0]) < current_day:
            ema_idx += 1
        while convert(rsi[rsi_idx][0]) < current_day:
            rsi_idx += 1
        while convert(aroon_up[aroon_up_idx][0]) < current_day:
            aroon_up_idx += 1
        while convert(aroon_down[aroon_down_idx][0]) < current_day:
            aroon_down_idx += 1
        while convert(stock_data[price_idx][0]) < current_day + K:
            price_idx += 1
        ema_data.append([value for _, value in ema[ema_idx-N:ema_idx]])
        rsi_data.append([value for _, value in rsi[rsi_idx-N:rsi_idx]])
        aroon_up_data.append([value for _, value in aroon_up[aroon_up_idx-N:aroon_up_idx]])
        aroon_down_data.append([value for _, value in aroon_down[aroon_down_idx-N:aroon_down_idx]])
        current_price_data.append(stock_data[price_idx - K][1])
        price_data.append(stock_data[price_idx][1])

        current_day += WINDOW_SIZE
    return ema_data, rsi_data, aroon_up_data, aroon_down_data, current_price_data, price_data



# 1. Convert to NumPy arrays
# Convert each list into numpy arrays
ema_data_np, rsi_data_np, aroon_up_data_np, aroon_down_data_np, current_price_data_np, price_data_np = map(np.array, generate_technical_data())

# Convert each technical indicator to have a shape of (num_samples, sequence_length, 1)
def reshape_data(data):
    return data.reshape(data.shape[0], data.shape[1], 1)

ema_data_3d = reshape_data(ema_data_np)
rsi_data_3d = reshape_data(rsi_data_np)
aroon_up_data_3d = reshape_data(aroon_up_data_np)
aroon_down_data_3d = reshape_data(aroon_down_data_np)

# Stack them along the feature axis
technical_data_3d = np.concatenate([ema_data_3d, rsi_data_3d, aroon_up_data_3d, aroon_down_data_3d], axis=2)

# Normalize data
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Reshape current_price_data for concatenation
current_price_data_np = current_price_data_np[:, np.newaxis]

# Apply the MinMaxScaler separately to each feature
for i in range(technical_data_3d.shape[2]):
    technical_data_3d[:, :, i] = scaler_x.fit_transform(technical_data_3d[:, :, i])

price_data_normalized = scaler_y.fit_transform(price_data_np.reshape(-1, 1))

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(technical_data_3d, price_data_normalized, test_size=0.2, shuffle=False)
X_price_train, X_price_test = train_test_split(current_price_data_np, test_size=0.2, shuffle=False)

# 2. Construct LSTM model with multiple inputs

# Technical indicators input branch
input_technical = Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm_tech = LSTM(50, return_sequences=True)(input_technical)
lstm_tech = Dropout(0.2)(lstm_tech)
lstm_tech = LSTM(50, return_sequences=True)(lstm_tech)
lstm_tech = Dropout(0.2)(lstm_tech)
lstm_tech = LSTM(50)(lstm_tech)
lstm_tech_out = Dropout(0.2)(lstm_tech)

# Current price input branch
input_price = Input(shape=(1,))
dense_price = Dense(16, activation='relu')(input_price)

# Merge the outputs of the two branches
merged = concatenate([lstm_tech_out, dense_price])

# Add a final dense layer
output = Dense(1)(merged)

# Compile the model
model = Model(inputs=[input_technical, input_price], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_train, X_price_train], y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)


# 6. Evaluate the model
y_pred = model.predict([X_test, X_price_test])
y_pred_original = scaler_y.inverse_transform(y_pred) # Convert back to original price scale
y_test_original = scaler_y.inverse_transform(y_test)

# Calculate MSE or any other error metric
mse = np.mean(np.square(y_pred_original - y_test_original))
print(f'Mean Squared Error on Test Data: {mse}')

y_train_pred = model.predict([X_train, X_price_train])
y_test_pred = model.predict([X_test, X_price_test])

# Directional accuracy
def directional_accuracy(y_true, y_pred):
    direction_true = np.sign(y_true[1:] - y_true[:-1])
    direction_pred = np.sign(y_pred[1:] - y_true[:-1])
    return np.mean(direction_true == direction_pred)

dir_acc = directional_accuracy(y_test_original, y_pred_original)
print(f'Directional Accuracy on Test Data: {dir_acc * 100:.2f}%')

# Combining data for plotting
actual_values = np.concatenate([y_train, y_test])
predicted_values = np.concatenate([y_train_pred, y_test_pred])

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

article_data = fetch_data_from_api("NEWS_SENTIMENT", SYMBOL, api_key=api_key)

income_statement = fetch_data_from_api("INCOME_STATEMENT", SYMBOL, api_key=api_key)
cash_flow = fetch_data_from_api("CASH_FLOW", SYMBOL, api_key=api_key)
