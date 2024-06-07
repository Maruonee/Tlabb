from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer, Qt

class StatusFrame(QGroupBox):
    def __init__(self, parent=None):
        super().__init__('고장진단상황', parent)
        self.setStyleSheet('background-color: white')
        self.status_label = QLabel('')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet('font-size: 48px')
        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        self.timer = QTimer(self)
        self.status_visible = True

    def update_status(self):
        if self.parent().machine_error == 0:
            if self.status_visible:
                self.status_label.setText('정상')
            else:
                self.status_label.setText('')
            self.status_label.setStyleSheet('color: green; font-size: 48px')
        elif self.parent().machine_error == 1:
            if self.status_visible:
                self.status_label.setText('고장 예측')
            else:
                self.status_label.setText('')
            self.status_label.setStyleSheet('color: orange; font-size: 48px')
        elif self.parent().machine_error == 2:
            if self.status_visible:
                self.status_label.setText('고장')
            else:
                self.status_label.setText('')
            self.status_label.setStyleSheet('color: red; font-size: 48px')

        self.status_visible = not self.status_visible