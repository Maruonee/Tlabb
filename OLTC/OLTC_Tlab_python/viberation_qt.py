import sys
import serial
import time
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QComboBox, QLineEdit, QMessageBox

class SerialTimeSender(QWidget):
    def __init__(self):
        super().__init__()

        # UI 요소 초기화
        self.initUI()

    def initUI(self):
        # 포트와 보드레이트 입력 필드 및 전송 버튼 생성
        self.port_label = QLabel("COM Port:")
        self.port_input = QLineEdit("COM19")

        self.baudrate_label = QLabel("Baud Rate:")
        self.baudrate_input = QLineEdit("19200")

        self.send_button = QPushButton("Send Current Time")
        self.send_button.clicked.connect(self.send_time)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.port_label)
        layout.addWidget(self.port_input)
        layout.addWidget(self.baudrate_label)
        layout.addWidget(self.baudrate_input)
        layout.addWidget(self.send_button)

        self.setLayout(layout)

        # 윈도우 설정
        self.setWindowTitle("Serial Time Sender")

    def send_time(self):
        port = self.port_input.text()
        baudrate = int(self.baudrate_input.text())

        try:
            # 시리얼 포트 설정
            ser = serial.Serial(port, baudrate, timeout=2)
            time.sleep(2)  # 포트가 열리고 안정화될 시간을 주기 위해 지연

            # 현재 시간 가져오기
            current_time = time.strftime("%Y%m%d%H%M%S")  # YYYYMMDDHHMMSS 형식

            # 아두이노에 시간 전송
            ser.write(current_time.encode('utf-8'))
            ser.flush()  # 버퍼 비우기

            # 성공 메시지
            QMessageBox.information(self, "Success", "Time sent successfully!")

        except serial.SerialException as e:
            QMessageBox.critical(self, "Serial Error", f"Serial error: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {e}")
        finally:
            if 'ser' in locals() and ser.is_open:
                ser.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SerialTimeSender()
    window.show()
    sys.exit(app.exec_())