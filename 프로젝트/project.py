import sys
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFileDialog, QLabel, QPushButton,
    QListWidget, QProgressBar, QTableWidget, QTableWidgetItem,
    QGroupBox, QVBoxLayout, QHBoxLayout, QSizePolicy
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

def read_image_unicode(path):
    try:
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError
        return img
    except:
        print(f"\u274c \uc774\ubbf8\uc9c0 \ub85c\ub4dc \uc2e4\ud328: {path}")
        return None

class DetectionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Target Detection UI")
        self.setGeometry(100, 100, 1200, 700)

        self.device = torch.device("cpu")
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(self.device)
        self.model.classifier = nn.Identity()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.thresholds = {
            "target_1": 0.89,
            "target_2": 0.87,
            "target_3": 0.85,
            "target_4": 0.75
        }

        self.target_labels = ["target_1", "target_2", "target_3", "target_4"]
        self.target_features = []
        self.cycle_images = []
        self.current_index = 0

        self.load_fixed_targets()

        main_layout = QHBoxLayout(self)

        left_box = QVBoxLayout()
        cyc_group = QGroupBox("Select Cycle Images (up to 100)")
        cyc_layout = QVBoxLayout()
        self.cyc_list = QListWidget()
        btn_cyc = QPushButton("Load Cycle Images")
        btn_cyc.clicked.connect(self.load_cycle)
        cyc_layout.addWidget(self.cyc_list)
        cyc_layout.addWidget(btn_cyc)
        cyc_group.setLayout(cyc_layout)
        left_box.addWidget(cyc_group)
        main_layout.addLayout(left_box, 1)

        center_box = QVBoxLayout()
        img_nav = QHBoxLayout()
        self.lbl_prev = QPushButton("\u25c0")
        self.lbl_prev.clicked.connect(self.show_prev)
        self.lbl_prev.setEnabled(False)
        self.lbl_image = QLabel("No Image")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_next = QPushButton("\u25b6")
        self.lbl_next.clicked.connect(self.show_next)
        self.lbl_next.setEnabled(False)
        img_nav.addWidget(self.lbl_prev)
        img_nav.addWidget(self.lbl_image, 1)
        img_nav.addWidget(self.lbl_next)
        self.lbl_sim = QLabel("Similarity: N/A")
        self.lbl_sim.setAlignment(Qt.AlignCenter)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        center_box.addLayout(img_nav, 5)
        center_box.addWidget(self.lbl_sim)
        center_box.addWidget(self.progress)
        main_layout.addLayout(center_box, 2)

        right_box = QVBoxLayout()
        record_header = QHBoxLayout()
        lbl_records = QLabel("Detection Records")
        lbl_records.setAlignment(Qt.AlignCenter)
        btn_reset = QPushButton("\ucd08\uae30\ud654")
        btn_reset.clicked.connect(self.reset_results)
        record_header.addWidget(lbl_records)
        record_header.addStretch()
        record_header.addWidget(btn_reset)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Time", "Image Name", "Result"])
        self.table.horizontalHeader().setStretchLastSection(True)
        right_box.addLayout(record_header)
        right_box.addWidget(self.table)
        main_layout.addLayout(right_box, 2)

    def load_fixed_targets(self):
        target_dir = os.path.join(os.getcwd(), "target_image")
        target_files = ["img_1.jpg", "img_2.jpg", "img_3.jpg", "img_4.jpg"]
        self.target_features.clear()
        for fname in target_files:
            path = os.path.join(target_dir, fname)
            vec = self.extract_feature_vector(path)
            if vec is not None:
                self.target_features.append(vec)
            else:
                print(f"\u274c Target feature vector load failed: {fname}")

    def extract_roi(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [cnt for cnt in contours if cv2.contourArea(cnt) > 20000]
        if not candidates:
            return None
        largest = max(candidates, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        masked = cv2.bitwise_and(img, img, mask=mask)
        x, y, w, h = cv2.boundingRect(largest)
        return masked[y:y+h, x:x+w]

    def extract_feature_vector(self, img_path):
        img = read_image_unicode(img_path)
        if img is None:
            return None
        roi = self.extract_roi(img)
        if roi is None:
            return None
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model(input_tensor).cpu().numpy().flatten()
        return vec

    def load_cycle(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Cycle Images", "", "Images (*.png *.jpg *.bmp)")
        if files:
            self.cycle_images = files[:100]
            self.cyc_list.clear()
            for f in self.cycle_images:
                self.cyc_list.addItem(os.path.basename(f))
            self.current_index = 0
            self.update_display_controls()

    def update_display_controls(self):
        has = len(self.cycle_images) > 0
        self.lbl_prev.setEnabled(has)
        self.lbl_next.setEnabled(has)
        self.show_image()

    def show_image(self):
        if not self.cycle_images:
            return
        path = self.cycle_images[self.current_index]
        pix = QPixmap(path).scaled(400, 300, Qt.KeepAspectRatio)
        self.lbl_image.setPixmap(pix)

        vec = self.extract_feature_vector(path)
        result = "Non-target"
        best_sim = 0.0
        if vec is not None and self.target_features:
            sims = [cosine_similarity([vec], [tv])[0][0] for tv in self.target_features]
            best_sim = max(sims)
            best_idx = sims.index(best_sim)
            best_label = self.target_labels[best_idx]
            threshold = self.thresholds.get(best_label, 0.85)
            if best_sim >= threshold:
                result = best_label

        self.lbl_sim.setText(f"Similarity: {best_sim:.2f}")
        total = len(self.cycle_images)
        self.progress.setValue(int((self.current_index+1)/total*100))
        now = datetime.now().strftime("%H:%M:%S")
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(now))
        self.table.setItem(row, 1, QTableWidgetItem(os.path.basename(path)))
        self.table.setItem(row, 2, QTableWidgetItem(result))

    def show_next(self):
        if self.current_index < len(self.cycle_images) - 1:
            self.current_index += 1
            self.show_image()

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def reset_results(self):
        self.table.setRowCount(0)
        self.progress.setValue(0)
        self.lbl_sim.setText("Similarity: N/A")
        self.lbl_image.setPixmap(QPixmap())
        self.lbl_image.setText("No Image")
        self.current_index = 0
        self.lbl_prev.setEnabled(False)
        self.lbl_next.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionUI()
    window.show()
    sys.exit(app.exec_())
