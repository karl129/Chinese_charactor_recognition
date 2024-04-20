import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import pytesseract
from PIL import Image
from src.inference import inference, load_model


class OCRWindow(QMainWindow):
    def __init__(self, inference, model):
        super().__init__()
        self.setWindowTitle("手写汉字识别系统")
        self.setGeometry(500, 300, 350, 380)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(63, 15, 224, 224)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid black")

        self.upload_button = QPushButton('上传图片', self)
        self.upload_button.setGeometry(50, 280, 100, 30)
        self.upload_button.clicked.connect(self.upload_image)

        self.recognize_button = QPushButton('识别文字', self)
        self.recognize_button.setGeometry(200, 280, 100, 30)
        self.recognize_button.clicked.connect(self.recognize_text)

        self.exit_button = QPushButton('退出', self)
        self.exit_button.setGeometry(125, 320, 100, 30)
        self.exit_button.clicked.connect(self.close)

        self.result_label = QLabel(self)
        self.result_label.setGeometry(63, 245, 224, 50)
        self.result_label.setAlignment(Qt.AlignCenter)  # 底部对齐，水平居中
        # self.result_label.setStyleSheet("border: 2px solid black")
        
        self.file_name = None
        self.inference = inference
        self.model = model

    def upload_image(self):
        self.file_name, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image files (*.jpg *.png)')
        print('图片已上传')
        if self.file_name is not None:
            pixmap = QPixmap(self.file_name)
            self.image_label.setPixmap(pixmap.scaled(224, 224, Qt.KeepAspectRatio))
            self.image_label.setScaledContents(True)

    def recognize_text(self):
        pixmap = self.image_label.pixmap()
        if pixmap:
            print('开始识别文字....')
            text = self.inference(self.file_name, self.model)
            self.show_result(text)
            print('识别完成!')
        else:
            self.show_result("请先上传图片！")

    def show_result(self, result):
        self.result_label.setText(result)
        font = QFont()
        font.setPointSize(20)
        self.result_label.setFont(font)


if __name__ == '__main__':
    model = load_model()
    app = QApplication(sys.argv)
    window = OCRWindow(inference, model)
    window.show()
    sys.exit(app.exec_())
