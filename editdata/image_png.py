import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage, QColor, QPen
from PyQt5.QtCore import Qt, QRectF, QPointF
from skimage.metrics import structural_similarity

class ImageAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('영상 질 비교(CNR, SNR, PSNR, SSIM)')

        # UI 세팅
        self.openButtonOriginal = QPushButton('원본 이미지 선택', self)
        self.openButtonCompare = QPushButton('비교 이미지 선택', self)
        self.openButtonProcessed = QPushButton('영상처리 이미지 선택', self)
        
        self.signalROIButtonOriginal = QPushButton('신호 ROI 설정', self)
        self.noiseROIButtonOriginal = QPushButton('노이즈 ROI 설정', self)
        self.copyROIButton = QPushButton('ROI 복사', self)
        
        #하단 결과
        self.calculateButton = QPushButton('계산', self)
        self.resultLabel = QLabel('결과', self)
        
        #보여주는 영상 크기
        self.imageViewOriginal = QGraphicsView(self)
        self.imageViewOriginal.setFixedSize(512, 512)
        self.imageViewCompare = QGraphicsView(self)
        self.imageViewCompare.setFixedSize(512, 512)
        self.imageViewProcessed = QGraphicsView(self)
        self.imageViewProcessed.setFixedSize(512, 512)
        
        self.sceneOriginal = QGraphicsScene(self)
        self.sceneCompare = QGraphicsScene(self)
        self.sceneProcessed = QGraphicsScene(self)
        
        self.imageViewOriginal.setScene(self.sceneOriginal)
        self.imageViewCompare.setScene(self.sceneCompare)
        self.imageViewProcessed.setScene(self.sceneProcessed)

        # 레이아웃 세팅
        hboxTop = QHBoxLayout()
        hboxTop.addWidget(self.openButtonOriginal)
        hboxTop.addWidget(self.openButtonCompare)
        hboxTop.addWidget(self.openButtonProcessed)

        hboxBottom = QHBoxLayout()
        hboxBottom.addWidget(self.signalROIButtonOriginal)
        hboxBottom.addWidget(self.noiseROIButtonOriginal)
        hboxBottom.addWidget(self.copyROIButton)

        hboxImages = QHBoxLayout()
        hboxImages.addWidget(self.imageViewOriginal)
        hboxImages.addWidget(self.imageViewCompare)
        hboxImages.addWidget(self.imageViewProcessed)

        vbox = QVBoxLayout()
        vbox.addLayout(hboxTop)
        vbox.addLayout(hboxBottom)
        vbox.addLayout(hboxImages)
        vbox.addWidget(self.calculateButton)
        vbox.addWidget(self.resultLabel)

        self.setLayout(vbox)

        # 연결
        self.openButtonOriginal.clicked.connect(self.openPNGOriginal)
        self.openButtonCompare.clicked.connect(self.openPNGCompare)
        self.openButtonProcessed.clicked.connect(self.openPNGProcessed)
        
        self.signalROIButtonOriginal.clicked.connect(lambda: self.startROISelection('signal'))
        self.noiseROIButtonOriginal.clicked.connect(lambda: self.startROISelection('noise'))
        self.copyROIButton.clicked.connect(self.copyROIsToCompare)
        self.calculateButton.clicked.connect(self.calculateMetrics)

        # 영상
        self.original_image = None
        self.compare_image = None
        self.processed_image = None
        self.qimageOriginal = None
        self.qimageCompare = None
        self.qimageProcessed = None
        self.signal_roi_original = None
        self.noise_roi_original = None
        self.signal_roi_compare = None
        self.noise_roi_compare = None
        self.current_roi_type = None
        self.is_selecting_original = True
        self.pixmap_itemOriginal = QGraphicsPixmapItem()
        self.pixmap_itemCompare = QGraphicsPixmapItem()
        self.pixmap_itemProcessed = QGraphicsPixmapItem()
        self.rect_itemSignalOriginal = None
        self.rect_itemNoiseOriginal = None
        self.rect_itemSignalCompare = None
        self.rect_itemNoiseCompare = None
        self.rect_itemSignalProcessed = None
        self.rect_itemNoiseProcessed = None
        self.start_pos = None

        self.sceneOriginal.addItem(self.pixmap_itemOriginal)
        self.sceneCompare.addItem(self.pixmap_itemCompare)
        self.sceneProcessed.addItem(self.pixmap_itemProcessed)
        
    # 원본 이미지 열기
    def openPNGOriginal(self):
        desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Original Image File', desktop_path, 'Image files (*.png *.jpg *.jpeg *.bmp)')
        if fname:
            self.original_image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if self.original_image is None:
                self.resultLabel.setText('이미지를 읽을 수 없습니다.')
            else:
                self.displayImage(self.original_image, 'original')
                self.original_filename = fname  # 파일 이름을 저장하는 부분 추가
    
    # 비교 이미지 열기
    def openPNGCompare(self):
        desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Compare Image File', desktop_path, 'Image files (*.png *.jpg *.jpeg *.bmp)')
        if fname:
            self.compare_image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if self.compare_image is None:
                self.resultLabel.setText('이미지를 읽을 수 없습니다.')
            else:
                self.displayImage(self.compare_image, 'compare')

    # 영상처리 이미지 열기
    def openPNGProcessed(self):
        desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Processed Image File', desktop_path, 'Image files (*.png *.jpg *.jpeg *.bmp)')
        if fname:
            self.processed_image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if self.processed_image is None:
                self.resultLabel.setText('이미지를 읽을 수 없습니다.')
            else:
                self.displayImage(self.processed_image, 'processed')
    
    # 영상 보여주기
    def displayImage(self, image, image_type):
        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        height, width = normalized_image.shape
        qimage = QImage(normalized_image.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
    
        if image_type == 'original':
            self.pixmap_itemOriginal.setPixmap(pixmap.scaled(512, 512, Qt.KeepAspectRatio))
        elif image_type == 'compare':
            self.pixmap_itemCompare.setPixmap(pixmap.scaled(512, 512, Qt.KeepAspectRatio))
        elif image_type == 'processed':
            self.pixmap_itemProcessed.setPixmap(pixmap.scaled(512, 512, Qt.KeepAspectRatio))
                
    # ROI 선택
    def startROISelection(self, roi_type):
        self.current_roi_type = roi_type
        self.is_selecting_original = True
        self.imageViewOriginal.setMouseTracking(True)
        self.imageViewOriginal.viewport().installEventFilter(self)

    # ROI 선택 방법
    def eventFilter(self, source, event):
        if self.is_selecting_original and source is self.imageViewOriginal.viewport():
            if event.type() == event.MouseButtonPress:
                self.start_pos = event.pos()
                return True
            elif event.type() == event.MouseMove:
                if self.start_pos:
                    self.updateROISelection(event.pos())
                return True
            elif event.type() == event.MouseButtonRelease:
                self.setROI(event.pos())
                self.start_pos = None
                return True
        return super().eventFilter(source, event)
        
    def updateROISelection(self, current_pos):
        scene_start_pos = self.imageViewOriginal.mapToScene(self.start_pos)
        scene_current_pos = self.imageViewOriginal.mapToScene(current_pos)
        x1, y1 = int(scene_start_pos.x()), int(scene_start_pos.y())
        x2, y2 = int(scene_current_pos.x()), int(scene_current_pos.y())

        rect = QRectF(QPointF(x1, y1), QPointF(x2, y2))
        pen = QPen(QColor(255, 0, 0, 100))
        pen.setWidth(2)
        brush = QColor(255, 0, 0, 50)

        if self.current_roi_type == 'signal':
            if self.rect_itemSignalOriginal:
                self.sceneOriginal.removeItem(self.rect_itemSignalOriginal)
            self.rect_itemSignalOriginal = self.sceneOriginal.addRect(rect, pen, brush)
        elif self.current_roi_type == 'noise':
            if self.rect_itemNoiseOriginal:
                self.sceneOriginal.removeItem(self.rect_itemNoiseOriginal)
            self.rect_itemNoiseOriginal = self.sceneOriginal.addRect(rect, pen, brush)

    def setROI(self, pos):
        scene_start_pos = self.imageViewOriginal.mapToScene(self.start_pos)
        scene_end_pos = self.imageViewOriginal.mapToScene(pos)
        x1, y1 = int(scene_start_pos.x()), int(scene_start_pos.y())
        x2, y2 = int(scene_end_pos.x()), int(scene_end_pos.y())

        if self.current_roi_type == 'signal':
            self.signal_roi_original = (x1, y1, x2, y2)
        elif self.current_roi_type == 'noise':
            self.noise_roi_original = (x1, y1, x2, y2)

        self.imageViewOriginal.setMouseTracking(False)
        self.imageViewOriginal.viewport().removeEventFilter(self)

    # ROI 복사
    def copyROIsToCompare(self):
        if self.signal_roi_original is None or self.noise_roi_original is None:
            self.resultLabel.setText('ROI가 필요함.')
            return

        self.signal_roi_compare = self.signal_roi_original
        self.noise_roi_compare = self.noise_roi_original
        self.signal_roi_processed = self.signal_roi_original
        self.noise_roi_processed = self.noise_roi_original

        # 신호 ROI 붙여넣기
        x1, y1, x2, y2 = self.signal_roi_compare
        rect_signal = QRectF(QPointF(x1, y1), QPointF(x2, y2))
        pen = QPen(QColor(255, 0, 0, 100))
        pen.setWidth(2)
        brush = QColor(255, 0, 0, 50)

        if self.rect_itemSignalCompare:
            self.sceneCompare.removeItem(self.rect_itemSignalCompare)
        self.rect_itemSignalCompare = self.sceneCompare.addRect(rect_signal, pen, brush)

        if self.rect_itemSignalProcessed:
            self.sceneProcessed.removeItem(self.rect_itemSignalProcessed)
        self.rect_itemSignalProcessed = self.sceneProcessed.addRect(rect_signal, pen, brush)

        # 노이즈 ROI 붙여넣기
        x1, y1, x2, y2 = self.noise_roi_compare
        rect_noise = QRectF(QPointF(x1, y1), QPointF(x2, y2))

        if self.rect_itemNoiseCompare:
            self.sceneCompare.removeItem(self.rect_itemNoiseCompare)
        self.rect_itemNoiseCompare = self.sceneCompare.addRect(rect_noise, pen, brush)

        if self.rect_itemNoiseProcessed:
            self.sceneProcessed.removeItem(self.rect_itemNoiseProcessed)
        self.rect_itemNoiseProcessed = self.sceneProcessed.addRect(rect_noise, pen, brush)
        
    # 수치 계산
    def calculateMetrics(self):
        if self.original_image is None or self.compare_image is None or self.processed_image is None:
            self.resultLabel.setText('영상불러오기가 안됨.')
            return
        if self.signal_roi_original is None or self.noise_roi_original is None:
            self.resultLabel.setText('ROI가 필요함.')
            return

        x1, y1, x2, y2 = self.signal_roi_original
        signal_roi_original_resized = self.original_image[y1:y2, x1:x2]
        signal_roi_compare_resized = self.compare_image[y1:y2, x1:x2]
        signal_roi_processed_resized = self.processed_image[y1:y2, x1:x2]

        x1, y1, x2, y2 = self.noise_roi_original
        noise_roi_original_resized = self.original_image[y1:y2, x1:x2]
        noise_roi_compare_resized = self.compare_image[y1:y2, x1:x2]
        noise_roi_processed_resized = self.processed_image[y1:y2, x1:x2]

        # SNR 계산출력
        snr_original = self.calculate_snr(signal_roi_original_resized, noise_roi_original_resized)
        snr_compare = self.calculate_snr(signal_roi_compare_resized, noise_roi_compare_resized)
        snr_processed = self.calculate_snr(signal_roi_processed_resized, noise_roi_processed_resized)
        
        # CNR 계산출력
        cnr_original = self.calculate_cnr(signal_roi_original_resized, noise_roi_original_resized)
        cnr_compare = self.calculate_cnr(signal_roi_compare_resized, noise_roi_compare_resized)
        cnr_processed = self.calculate_cnr(signal_roi_processed_resized, noise_roi_processed_resized)
        
        # PSNR 계산출력
        psnr_compare = self.calculate_psnr(self.original_image, self.compare_image)
        psnr_processed = self.calculate_psnr(self.original_image, self.processed_image)
        
        # SSIM 계산출력
        ssim_compare = self.calculate_ssim(self.original_image, self.compare_image)
        ssim_processed = self.calculate_ssim(self.original_image, self.processed_image)
        
        #결과 출력
        result_text = (f'결과\n신호 ROI 위치: {self.signal_roi_original}\n'
                       f'노이즈 ROI 위치: {self.noise_roi_original}\n'
                       f'SNR (원본): {snr_original:.4f}\n'
                       f'SNR (비교): {snr_compare:.4f}\n'
                       f'SNR (처리): {snr_processed:.4f}\n'
                       f'CNR (원본): {cnr_original:.4f}\n'
                       f'CNR (비교): {cnr_compare:.4f}\n'
                       f'CNR (처리): {cnr_processed:.4f}\n'
                       f'PSNR (비교): {psnr_compare:.4f}\n'
                       f'PSNR (처리): {psnr_processed:.4f}\n'
                       f'SSIM (비교): {ssim_compare:.4f}\n'
                       f'SSIM (처리): {ssim_processed:.4f}')
        self.resultLabel.setText(result_text)
        
        # 결과를 텍스트 파일로 저장
        with open(f"{self.original_filename}.txt", "w") as file:
            file.write(result_text)

    # SNR 계산 코드
    def calculate_snr(self, signal_roi, noise_roi):
        signal_mean = np.mean(signal_roi)
        noise_std = np.std(noise_roi)
        #분모 0일 경우
        if noise_std == 0:
            return np.inf
        else:
            snr = signal_mean / noise_std
            return snr
        
    # CNR 계산 코드
    def calculate_cnr(self, roi_a, roi_b):
        mean_a = np.mean(roi_a)
        mean_b = np.mean(roi_b)
        noise_std = np.std(roi_b) 
        #분모 0일 경우
        if noise_std == 0:
            return np.inf
        else:
            cnr = np.abs(mean_a - mean_b) / noise_std
            return cnr

    # PSNR 계산 코드
    def calculate_psnr(self, original_image, compared_image):
        mse = np.mean((original_image - compared_image) ** 2)
        if mse == 0:
            return np.inf
        max_pixel = np.max(original_image)
        psnr = (max_pixel ** 2) / mse
        return psnr

    # SSIM 계산 코드
    def calculate_ssim(self, original_image, compared_image):
        ssim_value = structural_similarity(original_image, compared_image, data_range=compared_image.max() - compared_image.min())
        return ssim_value
    
#코드 실행
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageAnalyzer()
    ex.show()
    sys.exit(app.exec_())
