from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QGroupBox

class ControlsFrame(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()

        self.start_button = QPushButton('고장 진단 시작', self)
        self.stop_button = QPushButton('고장 진단 중단', self)
        self.stop_button.setEnabled(False)

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)