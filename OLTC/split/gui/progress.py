from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QLabel, QProgressBar

class ProgressFrame(QGroupBox):
    def __init__(self, parent=None):
        super().__init__('진행률', parent)
        layout = QVBoxLayout()

        self.progress_label = QLabel('고장 진단 주기')
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)

        self.total_progress_label = QLabel('고장 진단 기간')
        self.total_progress_bar = QProgressBar(self)
        self.total_progress_bar.setMinimum(0)
        self.total_progress_bar.setValue(0)

        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.total_progress_label)
        layout.addWidget(self.total_progress_bar)
        self.setLayout(layout)

    def update_progress(self, value, maximum):
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)

    def update_total_progress(self, value, maximum):
        self.total_progress_bar.setMaximum(maximum)
        self.total_progress_bar.setValue(value)

    def reset_progress(self):
        self.progress_bar.setValue(0)
        self.total_progress_bar.setValue(0)