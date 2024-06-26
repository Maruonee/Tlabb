from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QLabel, QHBoxLayout, QPushButton  # PyQt5 위젯 및 레이아웃 클래스 가져오기
from PyQt5.QtCore import Qt  # PyQt5 코어 클래스 가져오기

class StatusWidget:
    def __init__(self, tap_position, tap_voltage, tap_up, tap_down):
        self.tap_position = tap_position  # 탭 위치 설정
        self.tap_voltage = tap_voltage  # 탭 전압 설정
        self.tap_up = tap_up  # 탭 업 상태 설정
        self.tap_down = tap_down  # 탭 다운 상태 설정
        self.layout = QVBoxLayout()  # 레이아웃 설정

        status_frame = QGroupBox('Diagnosis Results')  # 상태 프레임 설정
        status_frame.setStyleSheet('background-color: white')  # 프레임 스타일 설정
        status_layout = QVBoxLayout()  # 상태 레이아웃 설정
        self.status_label = QLabel('')  # 상태 라벨 설정
        self.status_label.setAlignment(Qt.AlignCenter)  # 라벨 중앙 정렬
        self.status_label.setStyleSheet('font-size: 24px')  # 라벨 스타일 설정
        status_layout.addWidget(self.status_label)  # 레이아웃에 라벨 추가
        status_frame.setLayout(status_layout)  # 프레임에 레이아웃 설정
        self.layout.addWidget(status_frame)  # 메인 레이아웃에 프레임 추가

        ecotap_status_frame = QGroupBox('ECOTAP Status')  # ECOTAP 상태 프레임 설정
        ecotap_status_layout = QVBoxLayout()  # ECOTAP 상태 레이아웃 설정
        tap_position_layout = QHBoxLayout()  # 탭 위치 레이아웃 설정
        self.tap_position_label = QLabel(f'Tap position: {self.tap_position}')  # 탭 위치 라벨 설정
        tap_position_layout.addWidget(self.tap_position_label)  # 탭 위치 레이아웃에 라벨 추가
        self.tap_up_button = QPushButton('Tap Up')  # 탭 업 버튼 설정
        self.tap_up_button.clicked.connect(self.tap_up_action)  # 탭 업 버튼 클릭 시 액션 연결
        self.tap_down_button = QPushButton('Tap Down')  # 탭 다운 버튼 설정
        self.tap_down_button.clicked.connect(self.tap_down_action)  # 탭 다운 버튼 클릭 시 액션 연결
        tap_position_layout.addWidget(self.tap_up_button)  # 탭 위치 레이아웃에 탭 업 버튼 추가
        tap_position_layout.addWidget(self.tap_down_button)  # 탭 위치 레이아웃에 탭 다운 버튼 추가
        ecotap_status_layout.addLayout(tap_position_layout)  # ECOTAP 상태 레이아웃에 탭 위치 레이아웃 추가
        self.tap_voltage_label = QLabel(f'Tap voltage: {self.tap_voltage}')  # 탭 전압 라벨 설정
        ecotap_status_layout.addWidget(self.tap_voltage_label)  # ECOTAP 상태 레이아웃에 탭 전압 라벨 추가
        ecotap_status_frame.setLayout(ecotap_status_layout)  # 프레임에 ECOTAP 상태 레이아웃 설정
        self.layout.addWidget(ecotap_status_frame)  # 메인 레이아웃에 ECOTAP 상태 프레임 추가

    def tap_up_action(self):
        self.tap_up = 999  # 탭 업 상태 변경
        self.tap_position_label.setText(f'Tap position: {self.tap_up}')  # 탭 위치 라벨 업데이트

    def tap_down_action(self):
        self.tap_down = 999  # 탭 다운 상태 변경
        self.tap_position_label.setText(f'Tap position: {self.tap_down}')  # 탭 위치 라벨 업데이트
