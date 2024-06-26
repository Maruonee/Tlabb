from PyQt5.QtCore import pyqtSignal, QObject  # PyQt5 코어 클래스 및 시그널 가져오기

class Logger(QObject):
    log_signal = pyqtSignal(str)  # 로그 시그널 정의

    def __init__(self):
        super().__init__()  # 부모 클래스 초기화

    def log(self, message):
        self.log_signal.emit(message)  # 로그 시그널 발생
