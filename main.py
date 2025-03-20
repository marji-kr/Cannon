import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QComboBox, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ORBApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI 설정
        self.setWindowTitle("ORB 검사 프로그램")
        self.setGeometry(100, 100, 800, 600)

        # 메인 메뉴 실행
        self.main_menu()

    def main_menu(self):
        """ 메인 메뉴 화면 (4번째 UI) """
        self.clear_ui()

        # 버튼 생성
        self.pass_fail_btn = QPushButton("Pass or Fail Process", self)
        self.new_model_btn = QPushButton("New Model Setting", self)
        self.modify_rois_btn = QPushButton("Modifying ROIs", self)

        # 버튼 동작 설정
        self.pass_fail_btn.clicked.connect(self.pass_fail_process)
        self.new_model_btn.clicked.connect(self.new_model_setting)
        self.modify_rois_btn.clicked.connect(self.modify_rois)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.pass_fail_btn)
        layout.addWidget(self.new_model_btn)
        layout.addWidget(self.modify_rois_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def pass_fail_process(self):
        """ Pass or Fail Process 화면 (1번째 UI) """
        self.clear_ui()

        # UI 요소 추가
        self.back_btn = QPushButton("Back", self)
        self.master_image_label = QLabel("Master Image", self)
        self.camera_label = QLabel("Camera", self)
        self.capture_label = QLabel("Capture Image", self)
        self.capture_process_btn = QPushButton("Capture and Process", self)
        self.threshold_table = QTableWidget(3, 4)
        self.threshold_table.setHorizontalHeaderLabels(["Index", "Score", "Threshold", "Pass/Fail"])

        # 버튼 동작 설정
        self.back_btn.clicked.connect(self.main_menu)
        self.capture_process_btn.clicked.connect(self.process_images)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.back_btn)
        layout.addWidget(self.master_image_label)
        layout.addWidget(self.camera_label)
        layout.addWidget(self.capture_label)
        layout.addWidget(self.capture_process_btn)
        layout.addWidget(self.threshold_table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def new_model_setting(self):
        """ New Model Setting 화면 (3번째 UI) """
        self.clear_ui()

        self.back_btn = QPushButton("Back", self)
        self.master_image_label = QLabel("Master Image", self)
        self.select_master_btn = QPushButton("Select Master Image", self)
        self.process_btn = QPushButton("Process", self)
        self.roi_selector = QComboBox(self)
        self.roi_selector.addItems([str(i) for i in range(1, 6)])

        # 버튼 동작 설정
        self.back_btn.clicked.connect(self.main_menu)
        self.select_master_btn.clicked.connect(self.load_master_image)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.back_btn)
        layout.addWidget(self.master_image_label)
        layout.addWidget(self.select_master_btn)
        layout.addWidget(self.roi_selector)
        layout.addWidget(self.process_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def modify_rois(self):
        """ Modifying ROIs 화면 (2번째 UI) """
        self.clear_ui()

        self.back_btn = QPushButton("Back", self)
        self.modify_btn = QPushButton("Modify")
        self.add_btn = QPushButton("Add")
        self.delete_btn = QPushButton("Delete")

        # 버튼 동작 설정
        self.back_btn.clicked.connect(self.main_menu)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.back_btn)
        layout.addWidget(self.modify_btn)
        layout.addWidget(self.add_btn)
        layout.addWidget(self.delete_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_master_image(self):
        """ 마스터 이미지 로드 """
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Master Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.master_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            pixmap = QPixmap(file_name).scaled(300, 300, Qt.KeepAspectRatio)
            self.master_image_label.setPixmap(pixmap)

    def process_images(self):
        """ ORB 기반 Pass/Fail 판별 """
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Test Image", "", "Images (*.png *.jpg *.jpeg)")
        if not file_name:
            return

        test_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        # ORB 알고리즘 실행
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(self.master_image, None)
        kp2, des2 = orb.detectAndCompute(test_image, None)

        # BFMatcher로 특징점 매칭
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # 매칭 개수 기반 Pass/Fail 판별
        match_threshold = 10
        status = "PASS" if len(matches) >= match_threshold else "FAIL"

        # 결과 테이블 업데이트
        row_position = self.threshold_table.rowCount()
        self.threshold_table.insertRow(row_position)
        self.threshold_table.setItem(row_position, 0, QTableWidgetItem(str(row_position + 1)))
        self.threshold_table.setItem(row_position, 1, QTableWidgetItem(str(len(matches))))
        self.threshold_table.setItem(row_position, 2, QTableWidgetItem(str(match_threshold)))
        self.threshold_table.setItem(row_position, 3, QTableWidgetItem(status))

    def clear_ui(self):
        """ 기존 UI 요소 제거 """
        if self.centralWidget() is None:
            self.setCentralWidget(QWidget())

        layout = self.centralWidget().layout()
        if layout is not None:
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().setParent(None)

# 실행
app = QApplication(sys.argv)
window = ORBApp()
window.show()
sys.exit(app.exec_())
