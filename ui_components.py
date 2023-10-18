import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
    QFrame, QLineEdit, QPushButton, QFormLayout, QTableWidget, QHeaderView, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QThreadPool, pyqtSlot

from constants import START_DATE, END_DATE, BATCH_SIZE, METRICS
from inference import TickerPredictorModel
from utils.worker import Worker

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

# Constants
from constants import DEFAULT_TEXT_INPUTS


class TickerPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.threadpool = QThreadPool()

    def setup_ui(self):
        """Initialize the UI components."""
        self.configure_window()
        main_layout = QVBoxLayout()

        # Create and set layouts
        main_layout.addLayout(self.create_title_section())
        main_layout.addLayout(self.create_main_content_section())

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)

        self.setCentralWidget(central_widget)

    def configure_window(self):
        """Set window configurations and styles."""
        self.setWindowTitle("ticker-predictor")
        self.setGeometry(100, 100, 800, 600)
        self.setFixedSize(1200, 800)

        with open("styles.css", "r") as f:
            self.setStyleSheet(f.read())

    def create_title_section(self) -> QVBoxLayout:
        """Create the title section."""
        title = QLabel("ticker-predictor", self)
        title.setProperty("title", True)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title_layout = QVBoxLayout()
        title_layout.addWidget(title)
        return title_layout

    def create_main_content_section(self) -> QHBoxLayout:
        """Create the main content section."""
        h_layout = QHBoxLayout()
        h_layout.addLayout(self.create_parameters_section(), stretch=2)
        h_layout.addWidget(self.create_divider())
        h_layout.addLayout(self.create_analytics_section(), stretch=8)
        return h_layout

    def create_parameters_section(self) -> QVBoxLayout:
        """Create the parameters section."""
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.create_section_label("parameters", self))
        left_layout.addLayout(self.create_form_section())
        left_layout.addStretch(1)
        self.train_btn = self.create_train_button()
        left_layout.addWidget(self.train_btn)
        return left_layout

    def create_form_section(self) -> QFormLayout:
        """Create the form input section."""
        form_layout = QFormLayout()
        self.text_inputs = {}

        fields = ["Ticker", "Training Split", "Epochs", "Stride", "Window Length", "Forecast Length"]
        for field in fields:
            label_widget = QLabel(f"{field}:")
            label_widget.setProperty("formLabel", True)
            input_field = QLineEdit(self)
            form_layout.addRow(label_widget, input_field)
            self.text_inputs[field] = input_field
            if field in DEFAULT_TEXT_INPUTS:
                self.text_inputs[field].setText(DEFAULT_TEXT_INPUTS[field])

        return form_layout

    def create_train_button(self) -> QPushButton:
        """Create the 'Train' button."""
        train_btn = QPushButton("Train", self)
        train_btn.clicked.connect(self.on_train_click)
        return train_btn

    def create_divider(self) -> QFrame:
        """Create a visual divider."""
        divider = QFrame(self)
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        divider.setProperty("divider", True)
        return divider

    def create_analytics_section(self) -> QVBoxLayout:
        """Create the analytics section."""
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.create_section_label("analytics", self))
        right_layout.addLayout(self.create_graph_and_metrics())
        return right_layout

    def create_graph_and_metrics(self) -> QVBoxLayout:
        """Create the graph and metrics section."""
        self.canvas = FigureCanvas(plt.figure())
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        self.metrics_table = self.create_metrics_table()

        graph_layout = QVBoxLayout()
        graph_layout.addWidget(self.canvas)
        graph_layout.addWidget(self.navi_toolbar)
        graph_layout.addWidget(self.metrics_table)

        return graph_layout

    def create_metrics_table(self) -> QTableWidget:
        """Create the metrics table."""
        metrics_table = QTableWidget(self)
        metrics_table.setColumnCount(2)
        metrics_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        metrics_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        metrics_table.setRowCount(8)
        return metrics_table

    @staticmethod
    def create_section_label(section_name, parent):
        label = QLabel(section_name, parent)
        label.setProperty("sectionTitle", True)
        if section_name == "ticker-predictor":
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        return label

    def on_train_click(self):
        """Handle the train button click."""
        self.train_btn.setDisabled(True)
        worker = Worker(
            self.run_model_instance, 
            ticker=self.text_inputs["Ticker"].text(), 
            training_split=self.text_inputs["Training Split"].text(), 
            epochs=self.text_inputs["Epochs"].text(), 
            stride=self.text_inputs["Stride"].text(), 
            window_length=self.text_inputs["Window Length"].text(), 
            forecast_length=self.text_inputs["Forecast Length"].text()
        )
        worker.signals.finished.connect(self.on_model_done)
        self.threadpool.start(worker)
    
    def run_model_instance(self, ticker: str, training_split: str, epochs: str, stride: str, window_length: str, forecast_length: str) -> dict:
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

        self.raw_dates = model.raw_dates
        return output
    
    @pyqtSlot(dict)
    def on_model_done(self, result):
        # Update the canvas with the new plot
        self.train_btn.setDisabled(False)
        self.canvas.figure.clear()
        ax = self.canvas.figure.subplots()
        
        # Convert raw_dates to a format matplotlib understands
        x_dates = [mdates.datestr2num(date) for date in self.raw_dates]
        
        # Plotting using x_dates as x-values
        actual_values = result["Actual Values"]
        predicted_values = result["Predicted Values"]
        
        ax.plot(x_dates, actual_values, label="Actual Values", color='blue')
        ax.plot(x_dates, predicted_values, label="Predicted Values", color='red', alpha=0.6)
        
        # Adjusting for train-test split
        train_length = int(len(x_dates) * float(self.text_inputs["Training Split"].text()))
        ax.axvline(x=x_dates[train_length], color='gray', linestyle='--', label='Train-Test Split')
        
        # Setting date format for x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Adjust as needed
        ax.figure.autofmt_xdate()  
        
        ax.set_title("Actual vs Predicted Stock Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.legend()
        
        self.canvas.draw()
        
        # Update the metrics labels
        for i, key in enumerate(METRICS):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(key))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(str(result[key])))