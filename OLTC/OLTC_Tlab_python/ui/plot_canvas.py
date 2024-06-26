import matplotlib.pyplot as plt  # matplotlib 가져오기
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # matplotlib FigureCanvas 가져오기

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, title=""):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)  # Figure와 Axes 생성
        super(PlotCanvas, self).__init__(fig)  # 부모 클래스 초기화
        self.setParent(parent)  # 부모 위젯 설정
        self.ax.set_title(title)  # 제목 설정
        self.ax.plot([])  # 빈 플롯 생성
