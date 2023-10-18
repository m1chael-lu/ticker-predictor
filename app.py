from ui_components import TickerPredictorApp
import sys
from PyQt6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = TickerPredictorApp()
    mainWin.show()
    sys.exit(app.exec())
