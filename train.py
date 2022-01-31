import argparse

import numpy as np # third party imports
import pandas as pd
from sklearn import preprocessing
from alpha_vantage.timeseries import TimeSeries
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras.models import Model
from keras import optimizers
import matplotlib.pyplot as plt


def structure_model():  
    # define two sets of inputs
    lstm_input = Input(shape=(window_size, 5), name='lstm_input')
    dense_input = Input(shape=(training_sma.shape[1],), name='tech_input')
    
    # the first branch operates on the first input
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)
    
    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)
    
    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')
    
    z = Dense(64, activation="relu", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    return Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)


def preprocess_data(symbol, window_size, forecast_length, training_split):
    data, meta_data = ts.get_daily(symbol, outputsize='full')

    data.to_csv(f'./{symbol}_daily.csv')

    window_size = 50
    forecast_length = 1
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



if __name__ == '__main__':
    symbol = "XOM"
    ts = TimeSeries(key="", output_format='pandas')
    window_size = 50
    forecast_length = 1
    training_split = 0.80

    # Processing data and structuring model
    training_x, training_y, testing_x, testing_y, training_sma, testing_sma, num_samples, regi_arr, sma, normalized_array = preprocess_data(symbol, window_size, forecast_length, training_split)
    testing_index = int(round(training_split * num_samples))
    model = structure_model()

    
    # Compiling and trianing model
    model.compile(optimizer="adam", loss='mse')
    model.fit(x=[training_x, training_sma], y=training_y, batch_size=32, epochs=30, shuffle=True, validation_split=0.1)

    predictions = model.predict([testing_x, testing_sma])
    predictions = predictions.reshape(len(predictions))
    mse = np.mean((predictions - testing_y)**2)
    print(mse)

    # Validating how accurate the up and down predictions are
    up_downs_prediction = np.zeros(len(predictions))
    up_downs_actual = np.zeros(len(testing_y))
    for i in range(len(predictions)):
        if predictions[i] > regi_arr[testing_index + i + window_size - forecast_length, 3]:
            up_downs_prediction[i] = 1
        if testing_y[i] > regi_arr[testing_index + i + window_size - forecast_length, 3]:
            up_downs_actual[i] = 1

    error = np.absolute(up_downs_actual - up_downs_prediction)
    ud_accuracy = 100 * (1 - np.mean(error))
    print(ud_accuracy)

    predictions_training = model.predict([training_x, training_sma])
    predictions_training = predictions_training.reshape(len(predictions_training))

    # Creating plot data
    plot_predict = np.concatenate((predictions_training, predictions))
    plot_actual = regi_arr[window_size + forecast_length:, 3]
    plot_predict = predictions[forecast_length:]
    plot_predict = predictions
    plot_actual = regi_arr[testing_index + window_size + forecast_length - 1:, 3]



    # Projection into the future
    projection_values = normalized_array[(regi_arr.shape[0] - window_size):, :].reshape((1, window_size, 5))
    projection_sma = (np.mean(regi_arr[-50:, 3]) - np.min(sma))/(np.max(sma) - np.min(sma))
    projection_sma = projection_sma.reshape((1, 1))
    future_projection = model.predict([projection_values, projection_sma])[0, 0]
    if future_projection > regi_arr[-1, 3]:
        print(symbol + " will likely be up in " + str(forecast_length) + " days with " + str(round(ud_accuracy)) + " percent confidence")
    else:
        print(symbol + " will likely be down in " + str(forecast_length) + " days with " + str(round(ud_accuracy)) + " percent confidence")

    # all_projections_values = np.zeros((forecast_length, window_size, 5))
    # for i in range(forecast_length):
    #     start_index = (regi_arr.shape[0] - window_size) - forecast_length + i + 1
    #     end_index = (regi_arr.shape[0]) - forecast_length + i + 1
    #     all_projections_values[i, :, :] = normalized_array[start_index:end_index, :]
    # all_projections = model.predict(all_projections_values).reshape(4)

    # plt.plot(np.concatenate((plot_predict[-50:], all_projections)))
    plt.plot(plot_predict[-1000:])
    plt.plot(plot_actual[-1000:])