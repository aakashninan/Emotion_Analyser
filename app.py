import os
# Force single-threaded initialization to prevent macOS mutex crash
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PYTHON = os.path.join(PROJECT_DIR, ".venv312", "bin", "python")

if __name__ == "__main__" and os.path.exists(PROJECT_PYTHON):
    current_python = os.path.abspath(sys.executable)
    if current_python != os.path.abspath(PROJECT_PYTHON):
        os.execv(PROJECT_PYTHON, [PROJECT_PYTHON, os.path.abspath(__file__), *sys.argv[1:]])

import cv2
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, 
    QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QFrame
)

model = None
model_error = None
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_model.h5")
FACE_CASCADE_PATH = os.path.join(PROJECT_DIR, "haarcascade_frontalface_default.xml")
face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def load_emotion_model():
    global model, model_error
    if model is not None or model_error is not None:
        return model

    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
    except Exception as error:
        model_error = str(error)
        print(f"Emotion model unavailable: {model_error}")
    return model

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
NEUTRAL_INDEX = emotions.index("Neutral")

class EmotionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Emotion Recognition Suite")
        self.resize(920, 740)
        
        # Modern Dark Theme Stylesheet (QSS)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
            }
            QLabel {
                color: #f8fafc;
                font-family: '-apple-system', 'Segoe UI', 'Roboto', sans-serif;
            }
            QPushButton {
                background-color: #334155;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 12px 18px;
                font-size: 13px;
                font-weight: 600;
                font-family: '-apple-system', 'Segoe UI', 'Roboto', sans-serif;
            }
            QPushButton:hover {
                background-color: #475569;
            }
            QPushButton:pressed {
                background-color: #1e293b;
            }
            #btnLive { background-color: #059669; }
            #btnLive:hover { background-color: #10b981; }
            
            #btnImg { background-color: #0284c7; }
            #btnImg:hover { background-color: #0ea5e9; }
            
            #btnVid { background-color: #7c3aed; }
            #btnVid:hover { background-color: #8b5cf6; }
            
            #btnStop { background-color: #e11d48; }
            #btnStop:hover { background-color: #f43f5e; }
            
            #videoFrame {
                background-color: #020617;
                border: 2px solid #1e293b;
                border-radius: 12px;
            }
        """)

        self.cap = None
        self.video_cap = None
        self.frame_counter = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_webcam_frame)

        # Main Central Widget & Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Header Section
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        
        title = QLabel("🎭 AI Emotion Recognition Suite", self)
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header_layout.addWidget(title)

        subtitle = QLabel("Real-time facial expression tracking via Computer Vision & Deep Learning", self)
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setStyleSheet("color: #94a3b8;")
        header_layout.addWidget(subtitle)
        
        layout.addLayout(header_layout)

        # Video/Image Display Container Frame
        self.frame_container = QFrame(self)
        self.frame_container.setObjectName("videoFrame")
        frame_layout = QVBoxLayout(self.frame_container)
        frame_layout.setContentsMargins(4, 4, 4, 4)

        self.display_label = QLabel(self)
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setText("Select an option below to begin analysis")
        self.display_label.setStyleSheet("color: #64748b; font-size: 14px; font-weight: 500; background: transparent;")
        self.display_label.setFixedSize(860, 480)
        frame_layout.addWidget(self.display_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.frame_container, alignment=Qt.AlignmentFlag.AlignCenter)

        # Action Buttons Layout Card
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        
        self.btn_live = QPushButton("🎥 Start Webcam", self)
        self.btn_live.setObjectName("btnLive")
        self.btn_live.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_live.clicked.connect(self.start_webcam)
        btn_layout.addWidget(self.btn_live)

        self.btn_img = QPushButton("🖼️ Upload Image", self)
        self.btn_img.setObjectName("btnImg")
        self.btn_img.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_img.clicked.connect(self.upload_image)
        btn_layout.addWidget(self.btn_img)

        self.btn_vid = QPushButton("🎞️ Upload Video", self)
        self.btn_vid.setObjectName("btnVid")
        self.btn_vid.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_vid.clicked.connect(self.upload_video)
        btn_layout.addWidget(self.btn_vid)

        self.btn_stop = QPushButton("🛑 Stop / Reset", self)
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.clicked.connect(self.stop_all)
        btn_layout.addWidget(self.btn_stop)

        layout.addLayout(btn_layout)

    def process_frame(self, frame):
        if frame is None:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            return frame

        face_inputs = []
        for startX, startY, width, height in faces:
            roi_gray = gray[startY:startY + height, startX:startX + width]
            roi_resized = cv2.resize(roi_gray, (48, 48)).astype(np.float32) / 255.0
            face_inputs.append(roi_resized.reshape(48, 48, 1))

        if load_emotion_model() is None:
            label_text = "Model unavailable"
            for startX, startY, width, height in faces:
                cv2.rectangle(frame, (startX, startY), (startX + width, startY + height), (0, 0, 255), 2)
                cv2.putText(frame, label_text, (startX, max(startY - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        predictions = model.predict(np.asarray(face_inputs), verbose=0)
        for (startX, startY, width, height), prediction in zip(faces, predictions):
            emotion_index = int(np.argmax(prediction))
            top_confidence = float(prediction[emotion_index])
            if top_confidence < 0.35:
                emotion_index = NEUTRAL_INDEX
            label = emotions[emotion_index]
            confidence = float(prediction[emotion_index]) * 100
            cv2.rectangle(frame, (startX, startY), (startX + width, startY + height), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.0f}%",
                (startX, max(startY - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return frame

    def start_webcam(self):
        self.stop_all()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open webcam.")
            return
        self.timer.start(30)

    def update_webcam_frame(self):
        source = self.cap or self.video_cap
        if source and source.isOpened():
            ret, frame = source.read()
            if ret:
                if source is self.cap:
                    frame = cv2.flip(frame, 1)
                self.frame_counter += 1
                if source is self.video_cap and self.frame_counter % 3 != 0:
                    self.display_image(frame)
                else:
                    self.display_image(self.process_frame(self.resize_frame(frame)))
            elif source is self.video_cap:
                self.stop_all()

    def upload_image(self):
        self.stop_all()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if file_path:
            file_bytes = np.fromfile(file_path, dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                QMessageBox.critical(self, "Error", "Could not read the selected image.")
                return
            self.display_image(self.process_frame(self.resize_frame(frame)))

    def upload_video(self):
        self.stop_all()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v)",
        )
        if file_path:
            self.video_cap = cv2.VideoCapture(file_path)
            if not self.video_cap.isOpened():
                self.video_cap.release()
                self.video_cap = None
                QMessageBox.critical(self, "Error", "Could not open the selected video.")
                return
            self.frame_counter = 0
            self.timer.start(30)

    @staticmethod
    def resize_frame(frame, max_width=1280):
        height, width = frame.shape[:2]
        if width <= max_width:
            return frame
        scale = max_width / width
        return cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)

    def display_image(self, img):
        if img is None:
            return
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_qt_format.scaled(860, 480, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.display_label.setPixmap(QPixmap.fromImage(p))

    def stop_all(self):
        self.timer.stop()
        self.frame_counter = 0
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.display_label.clear()
        self.display_label.setText("Select an option below to begin analysis")

    def closeEvent(self, event):
        self.stop_all()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec())