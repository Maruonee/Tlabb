import sys
from PyQt5.QtWidgets import QApplication
from data_collector_app import DataCollectorApp

machine_error = 1  # 0 = 정상 1 = 고장예측 2 = 고장

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataCollectorApp(machine_error)
    ex.show()
    sys.exit(app.exec_())