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

WINDOW_SIZE = 50
FORECAST_LENGTH = 1
TRAINING_SPLIT = 0.80
SYMBOL = "CVX"
EPOCHS = 30
BATCH_SIZE = 32
    
def fetch_data(symbol, api_key):
    """Fetch daily stock data using alpha_vantage."""

    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol, outputsize='full')
    data.to_csv(f'./{symbol}_daily.csv')
    return data

def structure_model(window_size, training_sma_shape_1):  
    """Construct the hybrid LSTM and technical indicator model."""

    # Define two sets of inputs
    lstm_input = Input(shape=(window_size, 5), name='lstm_input')
    dense_input = Input(shape=(training_sma_shape_1,), name='tech_input')
    
    # The first branch operates on the first input
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)
    
    # The second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)
    
    # Combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')
    
    z = Dense(64, activation="relu", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    return Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)


def preprocess_data(symbol, window_size, forecast_length, training_split, ts):
    """Preprocess stock data for LSTM and technical indicator models."""
    data, meta_data = ts.get_daily(symbol, outputsize='full')

    data.to_csv(f'./{symbol}_daily.csv')

    regi_df_features = data[["1. open", "2. high", "3. low", "4. close", "5. volume"]]
    regi_arr = np.array(regi_df_features)
    regi_arr = np.flip(regi_arr, 0)
    num_samples = regi_arr.shape[0] - window_size - forecast_length + 1

    formatted_y = np.zeros(num_samples)
    for i in range(num_samples):
        formatted_y[i] = regi_arr[i + window_size + forecast_length - 1, 3]

    # Simple Moving Average
    sma = []
    for i in range(num_samples):
        sma.append(np.mean(regi_arr[i:window_size + i, 3]))

    sma_arr = np.array(sma).reshape((len(sma), 1))
    sma_normalizer = preprocessing.MinMaxScaler()
    sma_normalized = sma_normalizer.fit_transform(sma_arr)


    # Normalize
    normalizer = preprocessing.MinMaxScaler()
    normalized_array = normalizer.fit_transform(regi_arr)

    # Format for LSTM
    normalized_formatted_x = np.zeros((num_samples, window_size, normalized_array.shape[1]))
    for i in range(normalized_array.shape[0] - window_size - forecast_length + 1):
        normalized_formatted_x[i, :, :5] = normalized_array[i:i+window_size, :]


    # Split between training and test
    training_split = 0.80
    testing_index = int(round(training_split * num_samples))

    training_x = normalized_formatted_x[:testing_index, :, :]
    training_sma = sma_normalized[:testing_index]
    testing_x = normalized_formatted_x[testing_index:, :, :]
    testing_sma = sma_normalized[testing_index:]

    training_y = formatted_y[:testing_index]
    testing_y = formatted_y[testing_index:]

    return training_x, training_y, testing_x, testing_y, training_sma, testing_sma, num_samples, regi_arr, sma, normalized_array

def calculate_accuracy(predictions, regi_arr, testing_index, window_size, forecast_length):
    """Calculate the up-down accuracy of the predictions."""
    up_downs_prediction = np.zeros(len(predictions))
    up_downs_actual = np.zeros(len(testing_y))
    for i in range(len(predictions)):
        if predictions[i] > regi_arr[testing_index + i + window_size - forecast_length, 3]:
            up_downs_prediction[i] = 1
        if testing_y[i] > regi_arr[testing_index + i + window_size - forecast_length, 3]:
            up_downs_actual[i] = 1

    error = np.absolute(up_downs_actual - up_downs_prediction)
    ud_accuracy = 100 * (1 - np.mean(error))
    return ud_accuracy


def predict_future(model, regi_arr, sma, forecast_length, ud_accuracy):
    """Predict and print stock direction for the next forecast_length days."""
    projection_values = normalized_array[(regi_arr.shape[0] - WINDOW_SIZE):, :].reshape((1, WINDOW_SIZE, 5))
    projection_sma = (np.mean(regi_arr[-50:, 3]) - np.min(sma))/(np.max(sma) - np.min(sma))
    projection_sma = projection_sma.reshape((1, 1))
    future_projection = model.predict([projection_values, projection_sma])[0, 0]
    if future_projection > regi_arr[-1, 3]:
        print(SYMBOL + " will likely be up in " + str(forecast_length) + " days with " + str(round(ud_accuracy)) + " percent confidence")
    else:
        print(SYMBOL + " will likely be down in " + str(forecast_length) + " days with " + str(round(ud_accuracy)) + " percent confidence")


if __name__ == '__main__':
    api_key = os.environ['ALPHA_VANTAGE_API_KEY']
    ts = TimeSeries(key=api_key, output_format='pandas')

    # Fetch and preprocess data
    training_x, training_y, testing_x, testing_y, training_sma, testing_sma, num_samples, regi_arr, sma, normalized_array = preprocess_data(
        SYMBOL, WINDOW_SIZE, FORECAST_LENGTH, TRAINING_SPLIT, ts)
    testing_index = int(round(TRAINING_SPLIT * num_samples))

    # Build, compile, and train the model
    model = structure_model(WINDOW_SIZE, training_sma.shape[1])
    model.compile(optimizer="adam", loss='mse')
    model.fit(x=[training_x, training_sma], y=training_y, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_split=0.1)

    # Evaluate the model
    predictions = model.predict([testing_x, testing_sma])
    mse = np.mean((predictions - testing_y) ** 2)
    print(f"Mean Squared Error: {mse}")

    accuracy = calculate_accuracy(predictions, regi_arr, testing_index, WINDOW_SIZE, FORECAST_LENGTH)
    print(f"Up/Down Accuracy: {accuracy}%")


    predictions_training = model.predict([training_x, training_sma])
    predictions_training = predictions_training.reshape(len(predictions_training))

    # Creating plot data
    plot_predict = np.concatenate((predictions_training, predictions.reshape(len(predictions))))
    plot_actual = regi_arr[WINDOW_SIZE + FORECAST_LENGTH:, 3]
    plot_predict = predictions[FORECAST_LENGTH:]
    plot_predict = predictions
    plot_actual = regi_arr[testing_index + WINDOW_SIZE + FORECAST_LENGTH - 1:, 3]


    # Projection into the future
    # all_projections_values = np.zeros((FORECAST_LENGTH, WINDOW_SIZE, 5))
    # for i in range(FORECAST_LENGTH):
    #     start_index = (regi_arr.shape[0] - WINDOW_SIZE) - FORECAST_LENGTH + i + 1
    #     end_index = (regi_arr.shape[0]) - FORECAST_LENGTH + i + 1
    #     all_projections_values[i, :, :] = normalized_array[start_index:end_index, :]
    # all_projections = model.predict(all_projections_values).reshape(4)

    # plt.plot(np.concatenate((plot_predict[-50:], all_projections)))
    plt.plot(plot_predict[-1000:])
    plt.plot(plot_actual[-1000:])
    plt.show()