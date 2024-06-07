from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QTextEdit

class LogOutputFrame(QGroupBox):
    def __init__(self, parent=None):
        super().__init__('로그 출력창', parent)
        layout = QVBoxLayout()

        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)

        layout.addWidget(self.log_output)
        self.setLayout(layout)

    def update_log(self, message):
        self.log_output.append(message)