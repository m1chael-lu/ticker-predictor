import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QFrame, QLineEdit, QPushButton, QFormLayout)
from PyQt6.QtCore import Qt, QRunnable, QThreadPool, pyqtSignal, QObject, pyqtSlot
from constants import START_DATE, END_DATE, BATCH_SIZE
from inference import TickerPredictorModel
import os
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class WorkerSignals(QObject):
    finished = pyqtSignal(dict)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        result = self.fn(*self.args, **self.kwargs)
        self.signals.finished.emit(result)

class TickerPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window configurations
        self.setWindowTitle("ticker-predictor")
        self.setGeometry(100, 100, 800, 600)
        self.setFixedSize(1200, 800)  # Fixed size

        # Load Stylesheet
        with open("styles.css", "r") as f:
            self.setStyleSheet(f.read())

        # Main layout
        main_layout = QVBoxLayout()

        # Title
        title = QLabel("ticker-predictor", self)
        title.setProperty("title", True)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # Parameters and Analytics sections
        params_label = QLabel("parameters", self)
        params_label.setProperty("sectionTitle", True)
        params_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        analytics_label = QLabel("analytics", self)
        analytics_label.setProperty("sectionTitle", True)
        analytics_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        divider = QFrame(self)
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        divider.setProperty("divider", True)

        # Text inputs for parameters
        form_layout = QFormLayout()

        # Add all the fields
        self.text_inputs = {}
        fields = ["Ticker", "Training Split", "Epochs", "Stride", "Window Length", "Forecast Length"]
        for field in fields:
            input_field = QLineEdit(self)
            form_layout.addRow(f"{field}:", input_field)
            self.text_inputs[field] = input_field
        
        self.text_inputs["Training Split"].setText("0.90")
        self.text_inputs["Epochs"].setText("55")
        self.text_inputs["Stride"].setText("3")
        self.text_inputs["Window Length"].setText("30")
        self.text_inputs["Forecast Length"].setText("1")

        # Train button
        train_btn = QPushButton("Train", self)
        train_btn.clicked.connect(self.on_train_click)
        
        # Horizontal layout for the two sections
        left_layout = QVBoxLayout()
        left_layout.addWidget(params_label)
        left_layout.addLayout(form_layout)
        left_layout.addStretch(1)
        left_layout.addWidget(train_btn)

        # Placeholder for the graph and metrics
        
        self.canvas = FigureCanvas(plt.figure())
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(analytics_label)
        right_layout.addWidget(self.canvas)
        self.metrics_labels = {}

        for key in ["MSE", "Accuracy", "Confusion Matrix", "Precision", "Recall", "F1 Score", "Future Price", "Current Price", "Dir Prediction"]:
            label = QLabel(self)
            right_layout.addWidget(label)
            self.metrics_labels[key] = label

        h_layout = QHBoxLayout()
        h_layout.addLayout(left_layout, stretch=2)
        h_layout.addWidget(divider)
        h_layout.addLayout(right_layout, stretch=8)
        main_layout.addLayout(h_layout, stretch=1)

        self.threadpool = QThreadPool()

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)

        self.setCentralWidget(central_widget)
    
    def on_train_click(self):
        worker = Worker(self.run_model_instance, 
                        ticker=self.text_inputs["Ticker"].text(), 
                        training_split=self.text_inputs["Training Split"].text(), 
                        epochs=self.text_inputs["Epochs"].text(), 
                        stride=self.text_inputs["Stride"].text(), 
                        window_length=self.text_inputs["Window Length"].text(), 
                        forecast_length=self.text_inputs["Forecast Length"].text())
        worker.signals.finished.connect(self.on_model_done)  # Connect the signal to the slot
        self.threadpool.start(worker)
    
    def run_model_instance(self, ticker, training_split, epochs, stride, window_length, forecast_length):
        # Use the values to create your TickerPredictorModel instance
        parameters = {
            "symbol": ticker,
            "training_split": float(training_split),
            "epochs": int(epochs),
            "window_size": int(stride),
            "n": int(window_length),
            "k": int(forecast_length),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "batch_size": BATCH_SIZE
        }
        api_key = os.environ['ALPHA_VANTAGE_API_KEY']
        model = TickerPredictorModel(parameters=parameters, api_key=api_key)
        model.fetch_data()
        model.generate_preprocess_data()
        model.construct_model()
        model.train()
        output = model.evaluate()
        output.update(model.prepare_plot())
        output.update(model.future_projection())
        return output
    
    @pyqtSlot(dict)
    def on_model_done(self, result):
        # Update the canvas with the new plot
        self.canvas.figure.clear()
        ax = self.canvas.figure.subplots()
        
        # Plotting the actual values
        actual_values = result["Actual Values"]
        predicted_values = result["Predicted Values"]
        
        ax.plot(actual_values, label="Actual Values", color='blue')
        ax.plot(predicted_values, label="Predicted Values", color='red', alpha=0.6)
        
        # You might need to adjust this if you're not passing train_length from the output
        train_length = len(actual_values) * 0.90  # Assuming 90% train-test split
        ax.axvline(x=train_length, color='gray', linestyle='--', label='Train-Test Split')
        
        ax.set_title("Actual vs Predicted Stock Prices")
        ax.set_xlabel("Time")
        ax.set_ylabel("Stock Price")
        ax.legend()
        
        self.canvas.draw()
        
        # Update the metrics labels
        for key, label in self.metrics_labels.items():
            label.setText(f"{key}: {result[key]}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = TickerPredictorApp()
    mainWin.show()
    sys.exit(app.exec())
