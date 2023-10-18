import os
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Input, concatenate
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from constants import income_statement_indicators
from keras.regularizers import l1_l2
from utils.data_extract import extract_technical_indicators, extract_prices, extract_income_statement, extract_cash_flow_statement
from constants import BATCH_SIZE, N
from utils.data_generation import generate_technical_data, generate_single_factor_data
from utils.helper import reshape_data
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from typing import Any, Dict, Union

class TickerPredictorModel:
    def __init__(self, parameters: Dict[str, Any], api_key: str):
        """Initialize the TickerPredictorModel with provided parameters and API key."""

        self.training_split = parameters["training_split"] 
        self.symbol = parameters["symbol"]
        if len(self.symbol) == 0:
            raise ValueError("Please provide a valid ticker symbol.")
        self.epochs = parameters["epochs"]
        self.batch_size = parameters["batch_size"]
        self.start_date = parameters["start_date"]
        self.end_date = parameters["end_date"]
        self.n = parameters["n"]
        self.window_size = parameters["window_size"]
        self.k = parameters["k"]
        self.api_key = api_key
    
    def fetch_data(self) -> None:
        """Fetch data from API and store relevant extracted information."""
        self.extracted_technical = extract_technical_indicators(self.api_key, self.symbol)
        self.extracted_income = extract_income_statement(self.api_key, self.symbol)
        self.stock_data = extract_prices(self.api_key, self.symbol)
        self.extracted_cash_flow = extract_cash_flow_statement(self.api_key, self.symbol)
        self.extracted_income.update(self.extracted_cash_flow)
    
    def generate_preprocess_data(self) -> None:
        """Generate and preprocess data for training."""
        
        # Generating data for technical indicators, price, and fundamental indicators
        generated_data, current_price_data, price_data, self.raw_dates = generate_technical_data(self.extracted_technical, self.stock_data, self.n)
        self.generated_fundamental = generate_single_factor_data(self.extracted_income)
        for key in generated_data: generated_data[key] = reshape_data(np.array(generated_data[key]))
        current_price_data_np, price_data_np = np.array(current_price_data), np.array(price_data)
        for key in self.generated_fundamental: self.generated_fundamental[key] = np.array(self.generated_fundamental[key])

        # Create a MinMaxScaler for each feature
        self.scalers_x = {}
        for key in generated_data:
            self.scalers_x[key] = MinMaxScaler(feature_range=(0, 1))

        # Stack them along the feature axis
        self.technical_data_order = [key for key in generated_data]
        for key in self.technical_data_order:
            generated_data[key][:, :, 0] = self.scalers_x[key].fit_transform(generated_data[key][:, :, 0]) 
        technical_data_3d = np.concatenate([generated_data[key] for key in self.technical_data_order], axis=2)

        # Reshape current_price_data for concatenation
        current_price_data_np = current_price_data_np[:, np.newaxis]
        price_data_normalized = price_data_np.reshape(-1, 1)

        # Store the order of the fundamental indicators for later use
        self.fundamental_order = [key for key in self.generated_fundamental]
        for key in self.fundamental_order:
            # Scaling by feature
            self.scalers_x[key] = MinMaxScaler(feature_range=(0, 1))
        for key in self.fundamental_order:
            self.generated_fundamental[key] = self.scalers_x[key].fit_transform(self.generated_fundamental[key].reshape(-1, 1))

        # Split the data into training and testing sets (based on parameter)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(technical_data_3d, price_data_normalized, test_size=1-self.training_split, shuffle=False)

        # Including single factor price into the generated fundamental single factor data
        self.generated_fundamental['currentPrice'] = price_data_normalized
        self.fundamental_order.append("currentPrice")

        # Creating train-test splits
        train_test_single_factor_splits = {}
        for key in self.generated_fundamental:
            train_test_single_factor_splits[key] = train_test_split(self.generated_fundamental[key], test_size=1-self.training_split, shuffle=False)

        # Concatenating all single factor data
        self.X_single_train = np.concatenate([train_test_single_factor_splits[key][0] for key in self.fundamental_order if type(key) == str], axis=1)
        self.X_single_test = np.concatenate([train_test_single_factor_splits[key][1] for key in self.fundamental_order if type(key) == str], axis=1)

    def construct_model(self) -> None:
        """Construct the hybrid LSTM model for stock price training/prediction."""

        # Technical indicators input branch
        input_technical = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        lstm_tech = LSTM(35, return_sequences=True, kernel_initializer='he_normal')(input_technical)
        lstm_tech = Dropout(0.2)(lstm_tech)
        lstm_tech = LSTM(35, return_sequences=True)(lstm_tech)
        lstm_tech = Dropout(0.2)(lstm_tech)
        lstm_tech = LSTM(35)(lstm_tech)
        lstm_tech_out = Dropout(0.2)(lstm_tech)

        # Single Factor Input Branch
        single_fact_input = Input(shape=(len(self.generated_fundamental.keys()),))
        dense_single = Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(single_fact_input)

        # Merge the outputs of the two branches
        merged = concatenate([lstm_tech_out, dense_single])

        # Add a final dense layer
        output = Dense(1)(merged)

        # Compile the model
        self.model = Model(inputs=[input_technical, single_fact_input], outputs=output)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train(self) -> None:
        """Train the model on the training dataset."""
        self.model.fit([self.X_train, self.X_single_train], self.y_train, epochs=self.epochs, batch_size=BATCH_SIZE, validation_split=0.1)
    
    def evaluate(self) -> Dict[str, Union[float, np.ndarray]]:
        """ Evaluate the model using MSE and Up/Down accuracy/"""

        # Creating model predictions
        output = {}
        y_pred = self.model.predict([self.X_test, self.X_single_test])
        y_pred_original = y_pred
        y_test_original = self.y_test

        # Calculate MSE
        mse = np.mean(np.square(y_pred_original - y_test_original))
        output["MSE"] = mse

        self.y_train_pred = self.model.predict([self.X_train, self.X_single_train])
        self.y_test_pred = self.model.predict([self.X_test, self.X_single_test])

        # Directional accuracy
        def directional_accuracy(y_true, y_pred):
            # Calculate the directional change for true and predicted values
            direction_true = np.sign(y_true[1:] - y_true[:-1])
            direction_pred = np.sign(y_pred[1:] - y_true[:-1])
            
            # Adjust zeros to 1 for both true and predicted directions
            direction_true[direction_true == 0] = 1
            direction_pred[direction_pred == 0] = 1
            
            return direction_true, direction_pred
        
        direction_true, direction_pred = directional_accuracy(y_test_original, y_pred_original)
        
        # Accuracy
        self.dir_acc = accuracy_score(direction_true, direction_pred)
        output["Accuracy"] = self.dir_acc

        # Confusion Matrix
        cm = confusion_matrix(direction_true, direction_pred)
        output["Confusion Matrix"] = cm
        print(cm)

        # Precision, Recall, F1 Score
        precision, recall, f1, _ = precision_recall_fscore_support(direction_true, direction_pred, average='binary')
        output["Precision"] = round(precision, 2)
        output["Recall"] = round(precision, 2)
        output["F1 Score"] = round(f1, 2)
        return output


    def prepare_plot(self) -> Dict[str, np.ndarray]:
        # Combining data for plotting
        output = {}
        output["Actual Values"] = np.concatenate([self.y_train, self.y_test])
        output["Predicted Values"] = np.concatenate([self.y_train_pred, self.y_test_pred])
        return output

    def plot_evaluation(self) -> None:
        """Visualize actual vs. predicted stock prices on a plot."""

        # Combining data for plotting
        actual_values = np.concatenate([self.y_train, self.y_test])
        predicted_values = np.concatenate([self.y_train_pred, self.y_test_pred])

        # Plotting the actual values
        plt.figure(figsize=(14, 7))
        plt.plot(actual_values, label="Actual Values", color='blue')

        # Plotting the predicted values
        plt.plot(predicted_values, label="Predicted Values", color='red', alpha=0.6)

        # Marking the train-test split point on the plot
        train_length = len(self.y_train)
        plt.axvline(x=train_length, color='gray', linestyle='--', label='Train-Test Split')

        # Labelling the plot
        plt.title("Actual vs Predicted Stock Prices")
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()
    
    def future_projection(self) -> Dict[str, Union[float, str]]:
        """Project future stock prices based on model predictions."""
        # Storing output for future projection, single_factor and multifactor as the single-sample input for the modle
        output = {}
        single_factor = np.zeros((1, 1 + len(self.extracted_income.keys())))
        multifactor = np.zeros((1, N, len(self.extracted_technical.keys())))

        # We iterate through the technical data order established before to ensure the same order of features
        # We extract the last self.n values (window size) for each technical indicator
        for i in range(len(self.technical_data_order)):
            key = self.technical_data_order[i]
            multifactor[0, :, i] = self.scalers_x[key].transform(np.array([value for _, value in self.extracted_technical[key][-self.n:]]).reshape((1, -1))).reshape(self.n)
        
        # We iterate through our original fundamental order, taking the most recent value that exists for each single factor
        for i in range(len(self.fundamental_order)):
            key = self.fundamental_order[i]
            if key == "currentPrice":
                single_factor[0, i] = np.array(self.stock_data[-1][1]).reshape((1, -1)).reshape(1)
            elif key in income_statement_indicators:
                single_factor[0, i] = self.scalers_x[key].transform(np.array(self.extracted_income[key][-1][1]).reshape((1, -1))).reshape(1)
            else:
                single_factor[0, i] = self.scalers_x[key].transform(np.array(self.extracted_cash_flow[key][-1][1]).reshape((1, -1))).reshape(1)
        
        # We make our prediction, and store the output 
        output["Future Price"] = float(self.model.predict([multifactor, single_factor]))
        output["Current Price"] = self.stock_data[-1][1]
        output["Dir Prediction"] = 'Up' if output["Future Price"] > self.stock_data[-1][1] else 'Down'
        return output