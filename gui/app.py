import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# Data structures for fast lookup
@dataclass
class Box:
    t: float
    b: Tuple[int, int, int, int]  # x1,y1,x2,y2
    cls: str
    track_id: int

class OverlayIndex:
    def __init__(self, bucket_ms: int = 33):
        self.bucket_ms = bucket_ms
        self.by_bucket: Dict[int, List[Box]] = {}

    def add(self, box: Box):
        key = int(round(box.t * 1000.0 / self.bucket_ms))
        self.by_bucket.setdefault(key, []).append(box)

    def get(self, t_sec: float) -> List[Box]:
        key = int(round(t_sec * 1000.0 / self.bucket_ms))
        return self.by_bucket.get(key, [])

class VideoPlayer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MSBL Analyzer Viewer")
        self.resize(1200, 800)

        # Central video label
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #202020;")
        self.setCentralWidget(self.video_label)

        # Bottom controls
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        self.open_btn = QtWidgets.QPushButton("Open Video…")
        self.play_btn = QtWidgets.QPushButton("Play")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.rate_combo = QtWidgets.QComboBox()
        self.rate_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x"]) 
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 1000)
        controls_layout.addWidget(self.open_btn)
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.pause_btn)
        controls_layout.addWidget(QtWidgets.QLabel("Rate"))
        controls_layout.addWidget(self.rate_combo)
        controls_layout.addWidget(self.slider, 1)
        dock = QtWidgets.QDockWidget("Transport", self)
        dock.setWidget(controls)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

        # Left events list
        self.events_list = QtWidgets.QListWidget()
        events_dock = QtWidgets.QDockWidget("Events", self)
        events_dock.setWidget(self.events_list)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, events_dock)

        # Overlay toggles
        toggle_widget = QtWidgets.QWidget()
        toggle_layout = QtWidgets.QVBoxLayout(toggle_widget)
        self.show_chars_cb = QtWidgets.QCheckBox("Characters", checked=True)
        self.show_ball_cb = QtWidgets.QCheckBox("Ball", checked=True)
        self.show_items_cb = QtWidgets.QCheckBox("Items", checked=True)
        self.show_ids_cb = QtWidgets.QCheckBox("Track IDs", checked=True)
        for w in [self.show_chars_cb, self.show_ball_cb, self.show_items_cb, self.show_ids_cb]:
            toggle_layout.addWidget(w)
        toggle_layout.addStretch(1)
        toggles_dock = QtWidgets.QDockWidget("Overlays", self)
        toggles_dock.setWidget(toggle_widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, toggles_dock)

        # Signals
        self.open_btn.clicked.connect(self.open_video)
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.slider.sliderPressed.connect(self._slider_pressed)
        self.slider.sliderReleased.connect(self._slider_released)
        self.events_list.itemDoubleClicked.connect(self._jump_to_event)

        # Timer for playback
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._on_timer)

        # State
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 30.0
        self.duration_ms = 0.0
        self.paused = True
        self.seeking_with_slider = False
        self.start_wall_time = 0.0
        self.start_video_ms = 0.0
        self.current_ms = 0.0

        # Overlays
        self.overlay_index = OverlayIndex(bucket_ms=33)
        self.events: List[dict] = []

    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video", os.getcwd(), "Video Files (*.mp4 *.mkv *.mov *.avi)")
        if not path:
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to open video")
            return
        self.fps = max(1e-3, self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.duration_ms = (frame_count / self.fps) * 1000.0 if frame_count else 0
        self.slider.setValue(0)
        self.current_ms = 0

        # Attempt to load analysis JSONs from same folder
        folder = os.path.dirname(path)
        self._load_json_overlays(folder)

        self._render_at_ms(0)

    def _load_json_overlays(self, folder: str):
        self.overlay_index = OverlayIndex(bucket_ms=max(10, int(1000 / max(1.0, self.fps))))
        self.events_list.clear()
        self.events = []
        # tracks.json
        tracks_path = os.path.join(folder, "tracks.json")
        if os.path.exists(tracks_path):
            try:
                with open(tracks_path, "r", encoding="utf-8") as f:
                    tracks = json.load(f)
                # Flatten to boxes
                for t in tracks:
                    cls = t.get("class", "")
                    tid = int(t.get("track_id", -1))
                    for entry in t.get("boxes", []):
                        ts = float(entry.get("t", 0.0))
                        b = entry.get("b") or entry.get("box")
                        if not b or len(b) != 4:
                            continue
                        box = Box(ts, (int(b[0]), int(b[1]), int(b[2]), int(b[3])), cls, tid)
                        self.overlay_index.add(box)
            except Exception as e:
                print("Failed to load tracks.json:", e)
        # events.json
        events_path = os.path.join(folder, "events.json")
        if os.path.exists(events_path):
            try:
                with open(events_path, "r", encoding="utf-8") as f:
                    self.events = json.load(f)
                for ev in self.events:
                    label = f"{ev.get('t', 0):.2f}s - {ev.get('type', 'event')} (conf {ev.get('confidence', 0):.2f})"
                    self.events_list.addItem(label)
            except Exception as e:
                print("Failed to load events.json:", e)

    def play(self):
        if not self.cap:
            return
        self.paused = False
        self.start_wall_time = QtCore.QElapsedTimer()
        self.start_wall_time.start()
        self.start_video_ms = self.current_ms
        rate = self._playback_rate()
        interval = max(1, int(1000 / (self.fps * rate)))
        self.timer.start(interval)

    def pause(self):
        self.paused = True
        self.timer.stop()

    def _on_timer(self):
        if not self.cap:
            return
        elapsed_ms = self.start_wall_time.elapsed() * self._playback_rate()
        target_ms = self.start_video_ms + elapsed_ms
        self._render_at_ms(target_ms)

    def _render_at_ms(self, target_ms: float):
        if not self.cap:
            return
        target_ms = max(0.0, min(self.duration_ms or target_ms, target_ms))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_ms = target_ms
        t_sec = target_ms / 1000.0
        overlay_boxes = self.overlay_index.get(t_sec)
        frame = self._draw_overlays(frame, overlay_boxes)
        self._display_frame(frame)
        if self.duration_ms > 0:
            self.slider.blockSignals(True)
            self.slider.setValue(int(1000 * self.current_ms / self.duration_ms))
            self.slider.blockSignals(False)

    def _draw_overlays(self, frame: np.ndarray, boxes: List[Box]) -> np.ndarray:
        show_chars = self.show_chars_cb.isChecked()
        show_ball = self.show_ball_cb.isChecked()
        show_items = self.show_items_cb.isChecked()
        show_ids = self.show_ids_cb.isChecked()
        for box in boxes:
            cls = (box.cls or "").lower()
            if cls.startswith("char") and not show_chars:
                continue
            if cls == "ball" and not show_ball:
                continue
            if cls.startswith("item") and not show_items:
                continue
            x1,y1,x2,y2 = box.b
            color = (0,255,0)
            if cls == "ball":
                color = (0, 200, 255)
            elif cls.startswith("item"):
                color = (255, 200, 0)
            elif cls.startswith("char"):
                color = (255, 0, 255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = cls
            if show_ids and box.track_id >= 0:
                label += f"#{box.track_id}"
            if label:
                cv2.putText(frame, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return frame

    def _display_frame(self, frame: np.ndarray):
        # Convert BGR to RGB and to QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        # Fit to label with aspect ratio
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        # Redraw last frame to fit new size
        if self.cap and self.current_ms >= 0:
            self._render_at_ms(self.current_ms)
        return super().resizeEvent(event)

    def _slider_pressed(self):
        self.seeking_with_slider = True

    def _slider_released(self):
        if not self.cap or not (self.duration_ms > 0):
            return
        pct = self.slider.value() / 1000.0
        self.current_ms = pct * self.duration_ms
        self._render_at_ms(self.current_ms)
        self.seeking_with_slider = False

    def _jump_to_event(self, item: QtWidgets.QListWidgetItem):
        idx = self.events_list.row(item)
        if 0 <= idx < len(self.events):
            t = float(self.events[idx].get("t", 0.0)) * 1000.0
            self.current_ms = t
            self._render_at_ms(self.current_ms)

    def _playback_rate(self) -> float:
        txt = self.rate_combo.currentText().replace("x", "")
        try:
            return float(txt)
        except Exception:
            return 1.0


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = VideoPlayer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
