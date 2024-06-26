import sys  # 시스템 관련 기능을 사용하기 위해 임포트
from PyQt5.QtWidgets import QApplication  # PyQt5의 QApplication 클래스 임포트
from ui.main_window import MainWindow  # MainWindow 클래스를 임포트

# 초기 변수 설정
machine_error = 0  # 기계 오류 상태
tap_position = 5  # 탭 위치
tap_voltage = 0  # 탭 전압
tap_up = 0  # 탭 업 상태
tap_down = 0  # 탭 다운 상태

if __name__ == '__main__':  # 스크립트가 직접 실행될 때만 실행
    app = QApplication(sys.argv)  # QApplication 인스턴스 생성
    main_window = MainWindow(machine_error, tap_position, tap_voltage, tap_up, tap_down)  # MainWindow 인스턴스 생성
    main_window.show()  # 메인 윈도우를 화면에 표시
    sys.exit(app.exec_())  # 이벤트 루프 실행 및 종료 시 종료 코드 반환

"""
project/
│
├── main.py
├── ui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── status_widget.py
│   ├── settings_widget.py
│   └── plot_canvas.py
├── workers/
│   ├── __init__.py
│   ├── recorder_worker.py
│   └── data_collector_worker.py
└── utils/
    ├── __init__.py
    ├── logger.py
    └── folder_helper.py

main.py: 애플리케이션의 진입점으로, QApplication 인스턴스를 생성하고 MainWindow를 초기화하여 보여줍니다.
ui/main_window.py: 메인 윈도우 클래스를 정의하며, UI의 주요 레이아웃과 요소들을 초기화하고 설정합니다.
ui/status_widget.py: 상태 위젯을 정의하며, 탭 위치 및 전압 정보를 표시합니다.
ui/settings_widget.py: 사용자 설정을 위한 위젯을 정의하며, 폴더 경로 선택, 진단 주기, 반복 횟수 등을 설정할 수 있습니다.
ui/plot_canvas.py: matplotlib을 사용하여 그래프를 그리는 캔버스를 정의합니다.
workers/recorder_worker.py: 사운드 데이터를 녹음하고 저장하는 작업을 수행하는 클래스를 정의합니다.
workers/data_collector_worker.py: 진동 데이터를 수집하고 저장하는 작업을 수행하는 클래스를 정의합니다.
utils/logger.py: 로그 메시지를 다른 컴포넌트로 전달하기 위해 사용되는 클래스를 정의합니다.
utils/folder_helper.py: 폴더를 생성하는 헬퍼 함수를 정의합니다.

"""