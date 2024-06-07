from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QFileDialog, QPushButton
from utils.utils import create_folder
import os

class SettingsFrame(QGroupBox):
    def __init__(self, parent=None):
        super().__init__('고장진단 주기 및 기간 설정', parent)
        
        self.savedir_label = QLabel('저장위치')
        self.savedir_input = QLineEdit(self)
        self.savedir_input.setText(os.path.join(os.path.expanduser("~"), 'downloads'))
        self.savedir_button = QPushButton('폴더선택', self)
        self.savedir_button.clicked.connect(self.browse_folder)

        self.duration_label = QLabel('고장진단 주기(초)')
        self.duration_input = QLineEdit(self)
        self.duration_input.setText('60')
        self.minute_checkbox = QCheckBox("1 min")
        self.minute_checkbox.stateChanged.connect(self.toggle_minute_checkbox)

        self.repeat_num_label = QLabel('고장진단 기간(주기반복횟수)')
        self.repeat_num_input = QLineEdit(self)
        self.repeat_num_input.setText('60')
        self.hour_checkbox = QCheckBox("1 hour")
        self.hour_checkbox.stateChanged.connect(self.toggle_hour_checkbox)
        self.month_checkbox = QCheckBox("1 month")
        self.month_checkbox.stateChanged.connect(self.toggle_month_checkbox)

        self.serial_port_label = QLabel('COM Port')
        self.serial_port_input = QLineEdit(self)
        self.serial_port_input.setText('COM3')
        self.baud_rate_label = QLabel('Arduino Baud Rate(bps)')
        self.baud_rate_input = QLineEdit(self)
        self.baud_rate_input.setText('19200')

        self.audio_samplerate_label = QLabel('Sampling Rate(Hz)')
        self.audio_samplerate_input = QLineEdit(self)
        self.audio_samplerate_input.setText('44100')

        self.exp_date_label = QLabel('날짜(YYMMDD)')
        self.exp_date_input = QLineEdit(self)
        self.exp_date_input.setText(datetime.now().strftime('%y%m%d'))
        self.exp_num_label = QLabel('번호')
        self.exp_num_input = QLineEdit(self)
        self.exp_num_input.setText('1')

        layout = QVBoxLayout()
        layout.addWidget(self.savedir_label)
        layout.addWidget(self.savedir_input)
        layout.addWidget(self.savedir_button)

        duration_hbox = QHBoxLayout()
        duration_hbox.addWidget(self.duration_input)
        duration_hbox.addWidget(self.minute_checkbox)
        layout.addWidget(self.duration_label)
        layout.addLayout(duration_hbox)

        repeat_num_layout = QHBoxLayout()
        repeat_num_layout.addWidget(self.repeat_num_input)
        repeat_num_layout.addWidget(self.hour_checkbox)
        repeat_num_layout.addWidget(self.month_checkbox)
        layout.addWidget(self.repeat_num_label)
        layout.addLayout(repeat_num_layout)

        layout.addWidget(self.serial_port_label)
        layout.addWidget(self.serial_port_input)
        layout.addWidget(self.baud_rate_label)
        layout.addWidget(self.baud_rate_input)

        layout.addWidget(self.audio_samplerate_label)
        layout.addWidget(self.audio_samplerate_input)

        layout.addWidget(self.exp_date_label)
        layout.addWidget(self.exp_date_input)
        layout.addWidget(self.exp_num_label)
        layout.addWidget(self.exp_num_input)

        self.setLayout(layout)

    def toggle_minute_checkbox(self):
        if self.minute_checkbox.isChecked():
            self.duration_input.setText('60')
            self.duration_input.setEnabled(False)
        else:
            self.duration_input.setEnabled(True)

    def toggle_hour_checkbox(self):
        if self.hour_checkbox.isChecked():
            self.repeat_num_input.setText('60')
            self.repeat_num_input.setEnabled(False)
        else:
            self.repeat_num_input.setEnabled(True)

    def toggle_month_checkbox(self):
        if self.month_checkbox.isChecked():
            self.repeat_num_input.setText('43200')
            self.repeat_num_input.setEnabled(False)
        else:
            self.repeat_num_input.setEnabled(True)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if folder:
            self.savedir_input.setText(folder)

    def get_settings(self):
        return {
            'savedir': self.savedir_input.text(),
            'duration': int(self.duration_input.text()),
            'baud_rate': int(self.baud_rate_input.text()),
            'serial_port': self.serial_port_input.text(),
            'repeat_num': int(self.repeat_num_input.text()),
            'exp_num': int(self.exp_num_input.text()),
            'exp_date': self.exp_date_input.text(),
            'audio_samplerate': int(self.audio_samplerate_input.text()),
            'sensor_recordings_folder_path': create_folder(self.savedir_input.text(), self.exp_date_input.text(), self.exp_num_input.text(), 'sensors'),
            'audio_recordings_folder_path': create_folder(self.savedir_input.text(), self.exp_date_input.text(), self.exp_num_input.text(), 'sound')
        }