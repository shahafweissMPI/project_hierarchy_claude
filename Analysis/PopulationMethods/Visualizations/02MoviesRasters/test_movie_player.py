"""
video_plot_sync.py
PyQt5 + PyQtGraph demo:
  * Plays myvideo.mp4
  * Slider shows / seeks time
  * Play‑pause button
"""
import sys, numpy as np, pathlib
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSlider, QFileDialog)


class VideoWithPlot(QWidget):
    def __init__(self, fps):
        super().__init__()
        self.setWindowTitle("PyQtGraph + Video sync")
        # ---------- Qt Multimedia ----------
        self.player = QMediaPlayer(self)
        self.player.setNotifyInterval(100)          # 100 ms updates
        video = QVideoWidget()
        self.player.setVideoOutput(video)
        # ---------- Transport controls ----------
        self.playBtn = QPushButton("▶")
        self.playBtn.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.player.positionChanged.connect(self.update_from_video)
        self.player.durationChanged.connect(self.slider.setMaximum)
        # ---------- Layout ----------
        controls = QHBoxLayout()
        controls.addWidget(self.playBtn)
        controls.addWidget(self.slider)
        main = QVBoxLayout(self)
        main.addWidget(video, stretch=5)
        main.addLayout(controls)
        # ---------- Open a test file ----------
        self.open_file(pathlib.Path("myvideo.mp4"))

    # ---- helpers ----
    def open_file(self, path):
        if not path.exists():
            path = QFileDialog.getOpenFileName(self, "Select MP4", str(pathlib.Path().cwd()),
                                               "Videos (*.mp4)")[0]
        if path:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(str(path))))

    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.playBtn.setText("▶")
        else:
            self.player.play()
            self.playBtn.setText("⏸")

    def update_from_video(self, ms):
        # keep slider in sync without feedback loops
        block = self.slider.blockSignals(True)
        self.slider.setValue(ms)
        self.slider.blockSignals(block)


if __name__ == "__main__":
    print("Now running: video_plot_sync.py")
    app = QApplication(sys.argv)
    # dummy signal: 10 minutes of 100 Hz samples
    fps = 30                                  # real video fps if known
    duration_s = 600
    win = VideoWithPlot(fps)
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())