import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
    QComboBox, QTableWidget, QTableWidgetItem, QSlider, QStackedWidget, QSpinBox, QFileDialog, QGridLayout
)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen

class MainMenu(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        btn_pass_fail = QPushButton("Pass or Fail Process")
        btn_pass_fail.setFixedSize(300, 60)
        btn_pass_fail.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))

        btn_new_model = QPushButton("New Model Setting")
        btn_new_model.setFixedSize(300, 60)
        btn_new_model.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))

        btn_modify_roi = QPushButton("Modifying ROIs")
        btn_modify_roi.setFixedSize(300, 60)
        btn_modify_roi.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))

        layout.addWidget(btn_pass_fail)
        layout.addWidget(btn_new_model)
        layout.addWidget(btn_modify_roi)

        self.setLayout(layout)

class SharedData:
    def __init__(self):
        self.master_image = None
        self.rois = []
        self.roi_images = []

class ROISelectableLabel(QLabel):
    def __init__(self, shared_data, num_rois_spinbox):
        super().__init__()
        self.shared_data = shared_data
        self.num_rois_spinbox = num_rois_spinbox
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.drawing = False
        self.selected_rois = []
        self.setMouseTracking(True)
        self.update_image()

    def update_image(self):
        if self.shared_data.master_image is not None:
            h, w = self.shared_data.master_image.shape
            bytes_per_line = w
            q_img = QImage(self.shared_data.master_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img).scaled(self.width(), self.height(), Qt.KeepAspectRatio)
            self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        # Modifying ROIs의 Add에서는 ROI 하나만 선택 가능 (num_rois_spinbox가 None일 때)
        if self.num_rois_spinbox is None:
            max_rois = 1
        else:
            max_rois = self.num_rois_spinbox.value()
        
        if event.button() == Qt.LeftButton and len(self.selected_rois) < max_rois:
            self.start_point = event.pos()
            self.drawing = True

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_point = event.pos()
            self.drawing = False
            rect = QRect(self.start_point, self.end_point).normalized()
            self.extract_roi(rect)
            self.selected_rois.append(rect)
            self.update()

    def extract_roi(self, rect):
        if self.shared_data.master_image is not None:
            label_w, label_h = self.width(), self.height()
            img_h, img_w = self.shared_data.master_image.shape
            scale_w = img_w / label_w
            scale_h = img_h / label_h
            x = int(rect.x() * scale_w)
            y = int(rect.y() * scale_h)
            w = int(rect.width() * scale_w)
            h = int(rect.height() * scale_h)
            roi_img = self.shared_data.master_image[y:y+h, x:x+w]
            self.shared_data.rois.append((x, y, w, h))
            self.shared_data.roi_images.append(roi_img)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        for rect in self.selected_rois:
            painter.drawRect(rect)

class ROISelectionWindow(QWidget):
    def __init__(self, shared_data, num_rois_spinbox, parent=None):
        super().__init__(parent)
        self.shared_data = shared_data
        self.num_rois_spinbox = num_rois_spinbox
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Select ROI")
        self.setGeometry(200, 200, 600, 400)

        layout = QVBoxLayout()
        self.label = ROISelectableLabel(self.shared_data, self.num_rois_spinbox)
        self.label.setFixedSize(500, 300)
        self.label.setStyleSheet("border: 2px solid black;")

        self.btn_save = QPushButton("Save ROIs")
        self.btn_save.clicked.connect(self.save_rois)

        layout.addWidget(self.label)
        layout.addWidget(self.btn_save)
        self.setLayout(layout)

    def save_rois(self):
        print("ROIs saved:", self.shared_data.rois)
        self.close()

class PassFailProcess(QWidget):
    def __init__(self, stacked_widget, shared_data):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.shared_data = shared_data
        self.initUI()

    def initUI(self):
        top_layout = QHBoxLayout()
        self.btn_back = QPushButton("Back")
        self.btn_back.setFixedSize(120, 40)
        self.btn_back.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.btn_test = QPushButton("Testing")
        self.btn_test.setFixedSize(120, 40)
        self.combo_file = QComboBox()
        self.combo_file.setFixedSize(200, 50)
        self.combo_file.addItem("파일 선택")
        top_layout.addWidget(self.btn_back)
        top_layout.addWidget(self.btn_test)
        top_layout.addStretch()
        top_layout.addWidget(self.combo_file)

        control_layout = QVBoxLayout()
        self.btn_get_started = QPushButton("Get Started")
        self.btn_get_started.setFixedSize(200, 50)
        self.btn_get_started.clicked.connect(self.toggleActivation)  # 상태 변경 함수 연결
        control_layout.addWidget(self.btn_get_started)

        self.roi_sliders = []
        roi_slider_layout = QVBoxLayout()
        for i in range(3):
            roi_row = QHBoxLayout()
            roi_label = QLabel(f"ROI #{i+1}")
            slider = QSlider(Qt.Horizontal)
            slider.setFixedSize(200, 30)
            slider.setEnabled(False)
            roi_row.addWidget(roi_label)
            roi_row.addWidget(slider)
            roi_slider_layout.addLayout(roi_row)
            self.roi_sliders.append(slider)

        self.btn_capture = QPushButton("Capture and Process")
        self.btn_capture.setFixedSize(200, 50)
        self.btn_capture.clicked.connect(self.process_image)
        control_layout.addWidget(self.btn_capture)
        control_layout.addStretch()

        image_layout = QHBoxLayout()
        self.label_camera = QLabel("Camera")
        self.label_camera.setFixedSize(250, 200)
        self.label_camera.setStyleSheet("border: 1px solid white;")
        self.label_master = QLabel("Master Image")
        self.label_master.setFixedSize(250, 200)
        self.label_master.setStyleSheet("border: 1px solid white;")
        image_layout.addWidget(self.label_camera)
        image_layout.addWidget(self.label_master)
        image_layout.addStretch()

        self.label_capture = QLabel("Capture Image")
        self.label_capture.setFixedSize(250, 100)
        self.label_capture.setStyleSheet("border: 1px solid white;")

        self.table = QTableWidget(3, 4)
        self.table.setFixedSize(500, 100)
        self.table.setHorizontalHeaderLabels(["Index", "Score", "Threshold", "Pass/Fail"])

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.label_capture)
        bottom_layout.addWidget(self.table)
        bottom_layout.addStretch()

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(image_layout)
        left_layout.addLayout(bottom_layout)
        main_layout.addLayout(left_layout)
        main_layout.addLayout(control_layout)

        final_layout = QVBoxLayout()
        final_layout.addLayout(top_layout)
        final_layout.addLayout(main_layout)
        self.setLayout(final_layout)

    def toggleActivation(self):
        if self.btn_get_started.text() == "Get Started":
            self.btn_get_started.setText("Activated")
            self.load_camera_image()
        else:
            self.btn_get_started.setText("Get Started")

    def load_camera_image(self):
        image_path = r"C:/Users/shjun/Desktop/학교자료/한양대4-1/머신러닝/2조 미니 프로젝트/데이터/defocused_blurred.jpg"

        try:
            image_data = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                self.display_image(self.label_camera, img)
                print("✅ Camera image loaded successfully!")
            else:
                print(f"⚠️ Error loading image: {image_path}")

        except Exception as e:
            print(f"❌ Failed to load image: {str(e)}")

    def process_image(self):
        """ Capture and Process 버튼 클릭 시 실행되는 기능 """
        image_path = r"C:/Users/shjun/Desktop/학교자료/한양대4-1/머신러닝/2조 미니 프로젝트/데이터/defocused_blurred.jpg"
        
        # Step 1: 카메라 이미지 불러오기
        try:
            image_data = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"⚠️ Error loading image: {image_path}")
                return
        except Exception as e:
            print(f"❌ Failed to load image: {str(e)}")
            return
        
        # Step 2: CLAHE 적용 (대비 보정)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(img)

        # Step 3: Deconvolution (역 컨볼루션) 적용
        deconv_img = self.simple_deconvolution(clahe_img)

        # Step 4: ORB 특징점 추출
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(deconv_img, None)

        # Step 5: ROI 이미지들과 비교하여 Pass/Fail 판정
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        threshold = 100  # Match Distance 기준 값
        results = []  # 결과 저장 리스트

        for idx, roi in enumerate(self.shared_data.roi_images):
            kp2, des2 = orb.detectAndCompute(roi, None)

            if des1 is None or des2 is None:
                score = 9999
                result = "Fail"
            else:
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                score = np.mean([m.distance for m in matches[:20]])
                result = "Pass" if score < threshold else "Fail"

            results.append((idx + 1, score, threshold, result))

        # Step 6: 결과 테이블 업데이트
        self.update_result_table(results)

        # Step 7: 복원된 이미지 화면에 표시
        self.display_image(self.label_camera, deconv_img)

    def simple_deconvolution(self, img):
        """ 이미지 복원을 위한 역 컨볼루션 적용 """
        kernel = np.ones((3, 3), np.float32) / 9
        blurred = cv2.filter2D(img, -1, kernel)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    def update_result_table(self, results):
        """ Pass/Fail 결과를 테이블에 업데이트 """
        self.table.setRowCount(len(results))
        for row, (idx, score, threshold, result) in enumerate(results):
            self.table.setItem(row, 0, QTableWidgetItem(str(idx)))
            self.table.setItem(row, 1, QTableWidgetItem(f"{score:.2f}"))
            self.table.setItem(row, 2, QTableWidgetItem(str(threshold)))
            self.table.setItem(row, 3, QTableWidgetItem(result))

    def display_image(self, label, img):
        """ QLabel에 이미지 표시 """
        h, w = img.shape
        bytes_per_line = w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
    
    def showEvent(self, event):
        self.update_master_image()

    def update_master_image(self):
        if self.shared_data.master_image is not None and self.shared_data.roi_images:
            roi_grid = self.create_roi_grid()
            h, w = roi_grid.shape
            bytes_per_line = w
            q_img = QImage(roi_grid.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img).scaled(self.label_master.width(), self.label_master.height(), Qt.KeepAspectRatio)
            self.label_master.setPixmap(pixmap)

    def create_roi_grid(self):
        num_rois = len(self.shared_data.roi_images)
        grid_size = (2, 3)  # 2 rows, 3 columns
        max_h = max(roi.shape[0] for roi in self.shared_data.roi_images)
        max_w = max(roi.shape[1] for roi in self.shared_data.roi_images)
        resized_rois = [cv2.resize(roi, (max_w, max_h)) for roi in self.shared_data.roi_images]

        grid_image = np.ones((max_h * grid_size[0], max_w * grid_size[1]), dtype=np.uint8) * 255
        
        for idx, roi in enumerate(resized_rois):
            row, col = divmod(idx, grid_size[1])
            y, x = row * max_h, col * max_w
            grid_image[y:y+max_h, x:x+max_w] = roi
        
        return grid_image

class NewModelSetting(QWidget):
    def __init__(self, stacked_widget, shared_data):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.shared_data = shared_data
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        self.btn_back = QPushButton("Back")
        self.btn_back.setFixedSize(100, 40)
        self.btn_back.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        top_layout.addWidget(self.btn_back)
        top_layout.addStretch()

        self.label_master = QLabel("Master Image")
        self.label_master.setFixedSize(400, 250)
        self.label_master.setStyleSheet("border: 1px solid black;")
        self.label_master.setAlignment(Qt.AlignCenter)

        self.btn_select_image = QPushButton("Select Master Image")
        self.btn_select_image.setFixedSize(200, 40)
        self.btn_select_image.clicked.connect(self.upload_image)

        roi_layout = QHBoxLayout()
        self.label_num_roi = QLabel("Num of ROIs")
        self.label_num_roi.setFixedSize(120, 30)
        self.spin_num_roi = QSpinBox()
        self.spin_num_roi.setFixedSize(60, 30)
        self.spin_num_roi.setRange(1, 10)
        roi_layout.addWidget(self.label_num_roi)
        roi_layout.addWidget(self.spin_num_roi)
        roi_layout.addStretch()

        self.btn_process = QPushButton("Process")
        self.btn_process.setFixedSize(200, 50)
        self.btn_process.clicked.connect(self.open_roi_selection)

        main_layout.addLayout(top_layout)
        main_layout.addStretch()
        main_layout.addWidget(self.label_master, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.btn_select_image, alignment=Qt.AlignCenter)
        main_layout.addStretch()
        main_layout.addLayout(roi_layout)
        main_layout.addWidget(self.btn_process, alignment=Qt.AlignCenter)
        main_layout.addStretch()
        self.setLayout(main_layout)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            pixmap = QPixmap(file_path)
            self.label_master.setPixmap(pixmap.scaled(self.label_master.width(), self.label_master.height(), Qt.KeepAspectRatio))

            # OpenCV에서 파일이 제대로 로드되었는지 확인
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Error loading image: {file_path}")
                return
        
            self.shared_data.master_image = img
            print("✅ Master image loaded successfully!")

    def open_roi_selection(self):
        if self.shared_data.master_image is None:
            print("⚠️ No master image selected!")  # 디버깅 메시지 추가
            return

        self.roi_window = ROISelectionWindow(self.shared_data, self.spin_num_roi)
        self.roi_window.show()
        self.roi_window.label.update_image()  # ROI 창에서 마스터 이미지 업데이트

# Modifying ROIs 화면
class ModifyingROIs(QWidget):
    def __init__(self, stacked_widget, shared_data):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.shared_data = shared_data
        self.selected_index = -1
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Modifying ROIs")
        self.setGeometry(100, 100, 800, 600)

        top_layout = QHBoxLayout()
        self.btn_back = QPushButton("Back")
        self.btn_back.setFixedSize(100, 40)
        self.btn_back.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))

        self.combo_file = QComboBox()
        self.combo_file.setFixedSize(600, 40)
        self.combo_file.addItem("파일 선택")

        top_layout.addWidget(self.btn_back)
        top_layout.addWidget(self.combo_file)

        self.roi_layout = QGridLayout()
        self.roi_labels = []
        for i in range(2):
            for j in range(3):
                label = QLabel()
                label.setFixedSize(150, 150)
                label.setStyleSheet("border: 1px solid white;")
                label.mousePressEvent = lambda e, idx=len(self.roi_labels): self.select_roi(idx)
                self.roi_labels.append(label)
                self.roi_layout.addWidget(label, i, j)

        bottom_layout = QHBoxLayout()
        self.btn_modify = QPushButton("Modify")
        self.btn_add = QPushButton("Add")
        self.btn_delete = QPushButton("Delete")

        self.btn_modify.clicked.connect(self.modify_roi)
        self.btn_add.clicked.connect(self.add_roi)
        self.btn_delete.clicked.connect(self.delete_selected_roi)

        bottom_layout.addWidget(self.btn_modify)
        bottom_layout.addWidget(self.btn_add)
        bottom_layout.addWidget(self.btn_delete)

        final_layout = QVBoxLayout()
        final_layout.addLayout(top_layout)
        final_layout.addLayout(self.roi_layout)
        final_layout.addLayout(bottom_layout)

        self.setLayout(final_layout)

    def showEvent(self, event):
        self.update_roi_display()

    def update_roi_display(self):
        for idx, label in enumerate(self.roi_labels):
            if idx < len(self.shared_data.roi_images):
                roi = self.shared_data.roi_images[idx]
                if roi is not None:
                    h, w = roi.shape
                    bytes_per_line = w
                    q_img = QImage(roi.tobytes(), w, h, bytes_per_line, QImage.Format_Grayscale8)
                    pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
                    label.setPixmap(pixmap)
                    label.setStyleSheet("border: 2px solid red;" if idx == self.selected_index else "border: 1px solid white;")
            else:
                label.clear()
                label.setStyleSheet("border: 1px solid white;")

    def select_roi(self, idx):
        self.selected_index = idx if idx < len(self.shared_data.roi_images) else -1
        self.update_roi_display()

    def delete_selected_roi(self):
        if self.selected_index != -1 and self.selected_index < len(self.shared_data.roi_images):
            del self.shared_data.roi_images[self.selected_index]
            del self.shared_data.rois[self.selected_index]
            self.selected_index = -1
            self.update_roi_display()

    def add_roi(self):
        if self.shared_data.master_image is None:
            print("⚠️ No master image selected!")
            return

        self.roi_window = ROISelectionWindow(self.shared_data, None)
        self.roi_window.show()
        self.roi_window.label.update_image()
       
         # ROI 창이 닫힌 후 새로 추가된 ROI 좌표 출력
        def on_close(event):
            print("ROIs saved:", self.shared_data.rois)
            self.update_roi_display()
            event.accept()

        self.roi_window.closeEvent = on_close

    def modify_roi(self):
        # ROI가 선택되지 않은 경우 처리
        if self.selected_index == -1 or self.selected_index >= len(self.shared_data.rois):
            print("⚠️ No ROI selected to modify!")
            return

        # 선택한 ROI 삭제
        del self.shared_data.roi_images[self.selected_index]
        del self.shared_data.rois[self.selected_index]
        print(f"ROI #{self.selected_index + 1} removed. Select a new ROI.")

        # Modify에서는 새 ROI 하나만 추가 가능하도록 ROISelectionWindow 호출
        self.roi_window = ROISelectionWindow(self.shared_data, None)
        self.roi_window.show()
        self.roi_window.label.update_image()

        # ROI 창이 닫힐 때 좌표 저장 및 UI 갱신
        def on_close(event):
            print("ROIs saved:", self.shared_data.rois)
            self.update_roi_display()
            event.accept()

        self.roi_window.closeEvent = on_close

# Main execution
if __name__ == '__main__':
    app = QApplication(sys.argv)
    stacked_widget = QStackedWidget()
    shared_data = SharedData()

    stacked_widget.addWidget(MainMenu(stacked_widget))
    stacked_widget.addWidget(PassFailProcess(stacked_widget, shared_data))
    stacked_widget.addWidget(NewModelSetting(stacked_widget, shared_data))
    stacked_widget.addWidget(ModifyingROIs(stacked_widget, shared_data))

    stacked_widget.setCurrentIndex(0)
    stacked_widget.show()
    sys.exit(app.exec_()) 
