import sys
import os
import json
import zmq
import base64
import threading
import datetime
import shutil
import re
import subprocess
import tempfile
import glob

import hashlib
import threading
import numpy as np

import requests
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSpinBox, QTextEdit, 
    QTabWidget, QListWidget, QGraphicsView, QGraphicsScene, 
    QSplitter, QMessageBox, QGroupBox, QFormLayout, QLineEdit,
    QComboBox, QCheckBox, QListWidgetItem, QSlider, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QRectF, QSize, QTimer, QBuffer, QByteArray
from PySide6.QtGui import QPixmap, QImage, QWheelEvent, QMouseEvent, QGuiApplication

class CameraManager:
    """Singleton thread-safe wrapper for cv2.VideoCapture."""
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, device):
        with cls._lock:
            if not cls._instance or cls._instance.device != device:
                if cls._instance and cls._instance.cap.isOpened():
                    cls._instance.cap.release()
                cls._instance = super().__new__(cls)
             
                backend = cv2.CAP_ANY
                actual_device = device
             
                # Determine the best backend based on device string
                if isinstance(device, str):
                    if device.startswith("libcamera:"):
                        # CSI / Modern laptop cameras managed by libcamera
                        if hasattr(cv2, "CAP_LIBCAMERA"):
                            backend = cv2.CAP_LIBCAMERA
                        actual_device = int(device.split(":", 1)[1])
                    elif os.name == 'posix' and device.startswith("/dev/video"):
                        # Standard USB UVC webcams on Linux
                        backend = cv2.CAP_V4L2
             
                cls._instance.cap = cv2.VideoCapture(actual_device, backend)
                cls._instance.device = device
                if not cls._instance.cap.isOpened():
                    raise RuntimeError(f"Failed to open {device} (backend: {backend})")
            return cls._instance

    def read_frame(self):
        with self._lock:
            ret, frame = self.cap.read()
            return ret, frame

    def set_prop(self, prop, val):
        with self._lock:
            self.cap.set(prop, val)

    @classmethod
    def release(cls):
        with cls._lock:
            if cls._instance and cls._instance.cap.isOpened():
                cls._instance.cap.release()
                cls._instance = None


def _get_frame_cache_dir():
    """Determine cache directory, preferring RAM disk on Linux to minimize disk wear."""
    if os.path.exists("/dev/shm"):
        return os.path.join("/dev/shm", "rag_frame_cache")
    return tempfile.gettempdir()

def _extract_video_frame(metadata, cache_dir):
    """Extract a single frame from a video using ffmpeg."""
    video_path = metadata.get("source_video", "")
    fps = float(metadata.get("fps", 30.0))
    frame_file = metadata.get("frame_file", "")
    
    if not video_path or not os.path.exists(video_path):
        return ""
        
    # Parse frame number from filename (e.g., "frame_0059.jpg" -> 59)
    match = re.search(r'(\d+)', frame_file)
    frame_num = int(match.group(1)) if match else 0
    timestamp = frame_num / fps
    
    out_name = f"frame_{metadata.get('id', 'vid')}_{frame_num}.jpg"
    out_path = os.path.join(cache_dir, out_name)
    
    if os.path.exists(out_path):
        return out_path
        
    print(f"calling ffmpeg to extract a frame from {video_path}")
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
        "-frames:v", "1", "-q:v", "2", out_path
    ]
    
    try:
        # Gracefully handle stdio to prevent blocking the UI thread
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
    except subprocess.TimeoutExpired:
        print(f"[FFmpeg] Timeout extracting frame from {video_path}")
    except FileNotFoundError:
        print("[FFmpeg] ffmpeg executable not found in PATH.")
    except Exception as e:
        print(f"[FFmpeg] Error: {e}")
    return ""



def _parse_v4l2_controls(device):
    """Parses v4l2-ctl output to heuristically generate a list of control dictionaries."""
    try:
        result = subprocess.run(["v4l2-ctl", "--list-ctrls-menus", "-d", device], 
                                capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []
        
        controls = []
        current_ctrl = None
        
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
                
            # Match standard int/menu controls
            # e.g., "brightness 0x00980900 (int) : min=0 max=255 step=1 default=128 value=128"
            match = re.match(r'(\w+)\s+(?:0x[0-9a-fA-F]+\s+)?\((int|menu)\)\s*:\s*min=(-?\d+)\s+max=(-?\d+)(?:\s+step=(-?\d+))?\s+default=(-?\d+)\s+value=(-?\d+)(.*)', line)
            if match:
                name, ctrl_type, min_val, max_val, step_val, default_val, value_val, rest = match.groups()
                
                props = {}
                for m in re.finditer(r'(\w+)=([^ ]+)', rest or ""):
                    props[m.group(1)] = m.group(2)
                    
                current_ctrl = {
                    "name": name,
                    "type": ctrl_type,
                    "min": int(min_val),
                    "max": int(max_val),
                    "step": int(step_val) if step_val else 1,
                    "default": int(default_val),
                    "value": int(value_val),
                    "flags": props.get("flags", ""),
                    "menu_items": []
                }
                controls.append(current_ctrl)
                continue
                
            # Match bool controls
            # e.g., "focus_automatic_continuous 0x00980a25 (bool) : default=1 value=1"
            match_bool = re.match(r'(\w+)\s+(?:0x[0-9a-fA-F]+\s+)?\(bool\)\s*:\s*default=(-?\d+)\s+value=(-?\d+)(.*)', line)
            if match_bool:
                name, default_val, value_val, rest = match_bool.groups()
                props = {}
                for m in re.finditer(r'(\w+)=([^ ]+)', rest or ""):
                    props[m.group(1)] = m.group(2)
                    
                current_ctrl = {
                    "name": name,
                    "type": "bool",
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "default": int(default_val),
                    "value": int(value_val),
                    "flags": props.get("flags", ""),
                    "menu_items": []
                }
                controls.append(current_ctrl)
                continue
                
            # Match menu items
            # e.g., "0: Manual Mode"
            if current_ctrl and re.match(r'^\d+:', line):
                match_menu = re.match(r'(\d+):\s+(.*)', line)
                if match_menu:
                    current_ctrl["menu_items"].append((int(match_menu.group(1)), match_menu.group(2)))
                    
        return controls
    except Exception as e:
        print(f"[V4L2] Error parsing controls: {e}")
        return []

def _set_v4l2_control(device, name, value):
    """Sets a V4L2 control using v4l2-ctl directly."""
    try:
        subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl", f"{name}={value}"], 
                       capture_output=True, text=True, timeout=5)
    except Exception as e:
        print(f"[V4L2] Failed to set control {name}: {e}")

def _parse_v4l2_formats(device):
    """Parses v4l2-ctl --list-formats-ext output to get available pixel formats and frame sizes."""
    formats = []
    try:
        result = subprocess.run(["v4l2-ctl", "-d", device, "--list-formats-ext"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return formats
        
        current_format = None
        for line in result.stdout.splitlines():
            line = line.strip()
            # Match format line, e.g., "[0]: 'YUYV' (YUYV 4:2:2)"
            match_fmt = re.search(r"\[\d+\]:\s*'([^']+)'", line)
            if match_fmt:
                current_format = match_fmt.group(1)
                formats.append({"format": current_format, "sizes": []})
                continue
            
            match_size = re.search(r"Size\s*:\s*Discrete\s+(\d+x\d+)", line)
            if match_size and current_format:
                size_str = match_size.group(1)
                if size_str not in formats[-1]["sizes"]:
                    formats[-1]["sizes"].append(size_str)
    except Exception as e:
        print(f"[V4L2] Error parsing formats: {e}")
    return formats


# =====================================================================
# CLEAN ZMQ WORKER (RAG CLIENT)
# =====================================================================


class ZmqConfigThread(QThread):
    """Stateless worker for RAG configuration commands."""
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, payload, target_address, timeout_ms=5000):
        super().__init__()
        self.payload = payload
        self.target_address = target_address
        self.timeout_ms = timeout_ms

    def run(self):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        
        try:
            socket.connect(self.target_address)
            socket.send_json(self.payload)
            response = socket.recv_json()
            self.finished_signal.emit(response)
        except zmq.error.Again:
            self.error_signal.emit("Server timeout: No response received within the time limit.")
        except Exception as e:
            self.error_signal.emit(f"Network error: {str(e)}")
        finally:
            try:
                socket.close(linger=0)
            except Exception as e:
                print(f"[ZmqConfigThread] Socket close error: {e}")
            
            try:
                ctx.term()
            except Exception as e:
                print(f"[ZmqConfigThread] Context term error: {e}")

class ZmqSearchThread(QThread):
    """Stateless worker for RAG queries."""
    finished_signal = Signal(dict, str)
    error_signal = Signal(str, str)

    def __init__(self, text_query, image_path, image_b64, limit, search_type, target_address, request_id="local_gui", timeout_ms=60000):
        super().__init__()
        self.text_query = text_query
        self.image_path = image_path
        self.image_b64 = image_b64
        self.limit = limit
        self.search_type = search_type
        self.target_address = target_address
        self.request_id = request_id
        self.timeout_ms = timeout_ms

    def run(self):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        
        try:
            socket.connect(self.target_address)

            img_b64 = self.image_b64
            if not img_b64 and self.image_path and os.path.exists(self.image_path):
                with open(self.image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')

            payload = {
                "action": "search",
                "query": self.text_query, 
                "image_path": self.image_path,
                "image_base64": img_b64,
                "limit": self.limit,
                "search_type": self.search_type
            }
            
            socket.send_json(payload)
            response = socket.recv_json()
            self.finished_signal.emit(response, self.request_id)
                
        except zmq.error.Again:
            self.error_signal.emit("Search timeout: The server took too long to respond.", self.request_id)
        except Exception as e:
            self.error_signal.emit(f"Search error: {str(e)}", self.request_id)
        finally:
            try:
                socket.close(linger=0)
            except Exception as e:
                print(f"[ZmqSearchThread] Socket close error: {e}")
                
            try:
                ctx.term()
            except Exception as e:
                print(f"[ZmqSearchThread] Context term error: {e}")

# =====================================================================
# DIAGNOSTIC COMPONENTS (Preserved)
# =====================================================================
class ZoomableGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.pixmap_item = None

    def load_image(self, base64_data: str, path: str):
        self.scene.clear()
        pixmap = QPixmap()
        
        # 1. Try Base64 First
        if base64_data:
            image_data = base64.b64decode(base64_data)
            pixmap.loadFromData(image_data)
        # 2. Fallback to Local Path
        elif path and os.path.exists(path):
            pixmap.load(path)
        else:
            self.scene.addText(f"Image not available or missing locally:\n{path}")
            return

        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        """Enable mouse-wheel zoom centered on cursor"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        
        # Anchor under mouse allows zooming precisely where the cursor is pointing
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
            
        self.scale(zoom_factor, zoom_factor)

class DiagnosticWindow(QWidget):
    def __init__(self, json_data, active_query_text=""):
        super().__init__()
        self.setWindowTitle("RAG Search Results Diagnostic")
        self.resize(1100, 800)
        self.json_data = json_data
        self.active_query_text = active_query_text
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # TAB 1: Visual Display
        tab_visual = QWidget()
        v_layout = QHBoxLayout(tab_visual)
        
        main_splitter = QSplitter(Qt.Horizontal)
        self.list = QListWidget()
        
        # Right side: Vertical split for Image (top) and Payload (bottom)
        right_splitter = QSplitter(Qt.Vertical)
        self.viewer = ZoomableGraphicsView()
        
        self.payload_viewer = QTextEdit()
        self.payload_viewer.setReadOnly(True)
        # Set a monospaced font for better JSON/Text readability
        self.payload_viewer.setStyleSheet("font-family: monospace;")
        
        right_splitter.addWidget(self.viewer)
        right_splitter.addWidget(self.payload_viewer)
        
        main_splitter.addWidget(self.list)
        main_splitter.addWidget(right_splitter)
        
        # Fix #1: Apply sensible default widths/heights so nothing is minimized
        main_splitter.setSizes([300, 800]) 
        right_splitter.setSizes([500, 300])
        
        v_layout.addWidget(main_splitter)
        
        # TAB 2: File/Text Viewer
        self.file_viewer = QTextEdit()
        self.file_viewer.setReadOnly(True)
        self.tabs.addTab(tab_visual, "Visual Display")
        self.tabs.addTab(self.file_viewer, "File Viewer")
        
        # TAB 3: Export Gallery
        self.setup_export_gallery()
        
        # TAB 4: Raw JSON
        raw = QTextEdit()
        raw.setPlainText(json.dumps(json_data, indent=4, ensure_ascii=False))
        self.tabs.addTab(raw, "Raw JSON")

        # Prepare cache directory for video frames
        self._cache_dir = _get_frame_cache_dir()
        os.makedirs(self._cache_dir, exist_ok=True)

        # Populate Lists
        for i, res in enumerate(json_data.get("results", [])):
            self.list.addItem(f"[{i+1}] Dist: {res.get('distance', 0):.3f}")
            meta = res.get("metadata", {})
            
            data = {
                "b64": res.get("image_base64", ""),
                "path": res.get("image_path", ""),
                "text": res.get("text", ""),
                "metadata": meta,
                "extracted_path": ""
            }
            
            # Extract frame if it's a video result without an existing path
            if meta.get("type") == "video_frame" and not data["path"]:
                data["extracted_path"] = _extract_video_frame(meta, self._cache_dir)
                
            self.list.item(i).setData(Qt.UserRole, data)

            
        # Fix #2: Use currentItemChanged to support keyboard arrow keys
        self.list.currentItemChanged.connect(self.on_item_changed)
        self.populate_gallery()
        
        # Select first item by default if it exists
        if self.list.count() > 0:
            self.list.setCurrentRow(0)

    def on_item_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        if current is None:
            return
            
        data = current.data(Qt.UserRole)
        b64 = data.get("b64")
        # Prioritize extracted frame path, then fall back to original path
        path = data.get("extracted_path", "") or data.get("path", "")
        text = data.get("text", "")
        metadata = data.get("metadata", {})
        
        # Handle Image Rendering
        if b64 or (path and path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.webp'))):
            self.viewer.load_image(b64, path)
        else:
            self.viewer.scene.clear()
            
        # Fix #3: Format and render the text/metadata payload, decoding escaped unicode
        payload_display = "=== TEXT PAYLOAD ===\n"
        payload_display += text + "\n\n"
        payload_display += "=== METADATA ===\n"
        # ensure_ascii=False forces the UI to render proper characters instead of \uXXXX
        payload_display += json.dumps(metadata, indent=4, ensure_ascii=False)
        self.payload_viewer.setPlainText(payload_display)
            
        # Handle File Rendering for the secondary tab
        if path and path.lower().endswith(('.txt', '.json', '.md', '.csv', '.py', '.cpp', '.h')):
            if os.path.exists(path):
                content = ""
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(path, 'r', encoding='utf-16', errors='replace') as f:
                        content = f.read()
                self.file_viewer.setPlainText(content)
            else:
                self.file_viewer.setPlainText(f"File not found locally:\n{path}\n\nEmbedded Text snippet:\n{text}")

    def setup_export_gallery(self):
        tab_export = QWidget()
        e_layout = QVBoxLayout(tab_export)
        
        top_bar = QHBoxLayout()
        btn_sel_all = QPushButton("Select All")
        btn_desel_all = QPushButton("Deselect All")
        self.chk_auto = QCheckBox("Auto-organize saves by timestamp and query")
        self.chk_auto.setChecked(True)
        btn_export = QPushButton("Export Selected Images...")
        btn_export.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        
        btn_sel_all.clicked.connect(lambda: self.set_gallery_checks(Qt.Checked))
        btn_desel_all.clicked.connect(lambda: self.set_gallery_checks(Qt.Unchecked))
        btn_export.clicked.connect(self.export_images)
        
        top_bar.addWidget(btn_sel_all)
        top_bar.addWidget(btn_desel_all)
        top_bar.addWidget(self.chk_auto)
        top_bar.addStretch()
        top_bar.addWidget(btn_export)
        
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setIconSize(QSize(200, 200))
        self.gallery.setResizeMode(QListWidget.Adjust)
        self.gallery.setSpacing(10)
        
        e_layout.addLayout(top_bar)
        e_layout.addWidget(self.gallery)
        self.tabs.addTab(tab_export, "Image Gallery & Export")

    def populate_gallery(self):
        for i, res in enumerate(self.json_data.get("results", [])):
            b64 = res.get("image_base64", "")
            path = res.get("image_path", "")
            
            # Retrieve prepared data including extracted paths
            item_data = self.list.item(i).data(Qt.UserRole)
            extracted = item_data.get("extracted_path", "")
            if extracted:
                path = extracted
                
            if not b64 and not (path and path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.webp'))):
                continue # Skip non-images
                
            item = QListWidgetItem(os.path.basename(path) if path else f"Embedded_Image_{i+1}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            
            # Load Icon
            pixmap = QPixmap()
            if b64:
                pixmap.loadFromData(base64.b64decode(b64))
            elif path and os.path.exists(path):
                pixmap.load(path)
                
            item.setIcon(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            # Store both original and extracted paths for export
            item.setData(Qt.UserRole, {"b64": b64, "path": path, "extracted_path": extracted, "index": i})
            self.gallery.addItem(item)

    def set_gallery_checks(self, state):
        for i in range(self.gallery.count()):
            self.gallery.item(i).setCheckState(state)

    def export_images(self):
        target_dir = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not target_dir: return
        
        if self.chk_auto.isChecked():
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_q = "".join([c for c in self.active_query_text if c.isalnum() or c in " -_"]).strip()[:30]
            if not safe_q: safe_q = "image_search"
            
            target_dir = os.path.join(target_dir, f"rag_export_{ts}_{safe_q}")
            os.makedirs(target_dir, exist_ok=True)
            
        count = 0
        for i in range(self.gallery.count()):
            item = self.gallery.item(i)
            if item.checkState() == Qt.Checked:
                data = item.data(Qt.UserRole)
                b64 = data.get("b64")
                # Prioritize extracted path for video frames
                path = data.get("extracted_path", "") or data.get("path", "")
                
                # Naming fallback
                orig_name = os.path.basename(path) if path else f"extracted_image_{data['index']}.jpg"
                save_path = os.path.join(target_dir, orig_name)
                
                try:
                    if b64:
                        with open(save_path, "wb") as f:
                            f.write(base64.b64decode(b64))
                    elif path and os.path.exists(path):
                        shutil.copy2(path, save_path)
                    count += 1
                except Exception as e:
                    QMessageBox.warning(self, "Export Error", f"Failed to save {orig_name}: {e}")
                    
        QMessageBox.information(self, "Export Complete", f"Successfully exported {count} image(s) to:\n{target_dir}")



class ImageInputWidget(QLabel):
    image_loaded = Signal(str, str) # path, base64_data

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Click, Drop, or Ctrl+V to select image")
        self.setStyleSheet("border: 2px dashed #aaa; padding: 10px; background: #f9f9f9; color: #555; min-height: 60px;")
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
            )
            if file_path:
                self.load_image_from_path(file_path)
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.load_image_from_path(file_path)
            event.acceptProposedAction()

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_V:
            clipboard = QGuiApplication.clipboard()
            mime_data = clipboard.mimeData()
            
            # Handle raw image buffer (e.g., screenshot)
            if mime_data.hasImage():
                pixmap = clipboard.pixmap()
                if not pixmap.isNull():
                    byte_array = QByteArray()
                    buffer = QBuffer(byte_array)
                    buffer.open(QBuffer.WriteOnly)
                    pixmap.save(buffer, "PNG")
                    self.load_image_from_bytes(byte_array.data(), "clipboard.png")
            # Handle copied file reference from OS file manager
            elif mime_data.hasUrls():
                urls = mime_data.urls()
                if urls and urls[0].isLocalFile():
                    self.load_image_from_path(urls[0].toLocalFile())
        else:
            super().keyPressEvent(event)

    def load_image_from_path(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.set_pixmap_scaled(pixmap)
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            self.image_loaded.emit(path, b64)
        else:
            self.setText("Failed to load image.")

    def load_image_from_bytes(self, image_bytes: bytes, source_name: str = "capture.jpg"):
        pixmap = QPixmap()
        if pixmap.loadFromData(image_bytes):
            self.set_pixmap_scaled(pixmap)
            b64 = base64.b64encode(image_bytes).decode('utf-8')
            self.image_loaded.emit(source_name, b64)
            return True
        else:
            self.setText("Failed to load image.")
            return False

    def set_pixmap_scaled(self, pixmap):
        scaled_pixmap = pixmap.scaled(
            self.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)


# =====================================================================
# REFINED RAG FRONT-END
# =====================================================================

class CameraDiscoveryThread(QThread):
    """Background worker to scan for cameras without blocking the UI."""
    finished_signal = Signal(list)

    def run(self):
        devices = []
        if os.name == "posix":
            try:
                # Increased timeout and relies on v4l2-ctl to prevent OpenCV FFMPEG spam
                out = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True, timeout=5)
                current_name = ""
                for line in out.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("/dev/video"):
                        # Only add /dev/video devices, skip /dev/media or others
                        devices.append((stripped, current_name or stripped))
                    elif stripped and not stripped.startswith("/dev/"):
                        # It's a camera name
                        current_name = stripped
            except Exception:
                # Fallback: test each /dev/video* with CAP_V4L2, suppressing OpenCV spam
                import sys as _sys
                old_stdout = _sys.stdout
                old_stderr = _sys.stderr
                _sys.stdout = open(os.devnull, 'w')
                _sys.stderr = open(os.devnull, 'w')
                try:
                    for d in sorted(glob.glob("/dev/video*")):
                        cap = cv2.VideoCapture(d, cv2.CAP_V4L2)
                        if cap.isOpened():
                            devices.append((d, d))
                            cap.release()
                finally:
                    _sys.stdout.close()
                    _sys.stderr.close()
                    _sys.stdout = old_stdout
                    _sys.stderr = old_stderr
                 
        # Libcamera discovery (CSI / Modern laptop cameras)
        try:
            libcamera_cmd = None
            if shutil.which("rpicam-vid"):
                libcamera_cmd = ["rpicam-vid", "--list-cameras"]
            elif shutil.which("libcamera-vid"):
                libcamera_cmd = ["libcamera-vid", "--list-cameras"]
         
            if libcamera_cmd:
                out = subprocess.check_output(libcamera_cmd, text=True, timeout=5)
                # Parse output like: "0 : imx219 [/base/...]"
                for match in re.finditer(r"(\d+)\s*:\s*([^\[]+)\[([^\]]+)\]", out):
                    idx = match.group(1)
                    name = match.group(2).strip()
                    # Use a special prefix so CameraManager knows to use CAP_LIBCAMERA
                    devices.append((f"libcamera:{idx}", f"[CSI/libcamera] {name} ({idx})"))
        except Exception as e:
            print(f"[Discovery] libcamera enumeration failed or not available: {e}")

        # Fallback for Windows/macOS or if no V4L2/libcamera devices found
        if not devices:
            import sys as _sys
            old_stdout = _sys.stdout
            old_stderr = _sys.stderr
            _sys.stdout = open(os.devnull, 'w')
            _sys.stderr = open(os.devnull, 'w')
            try:
                for i in range(16):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        devices.append((i, f"Camera Index {i}"))
                        cap.release()
            finally:
                _sys.stdout.close()
                _sys.stderr.close()
                _sys.stdout = old_stdout
                _sys.stderr = old_stderr
                    
        self.finished_signal.emit(devices)

class UvcCameraControlsWindow(QWidget):
    def __init__(self, camera_device):
        super().__init__()
        self.setWindowTitle(f"V4L2 Camera Controls - {camera_device}")
        self.resize(800, 600)
        self.camera_device = camera_device
        
        try:
            self.cam_mgr = CameraManager.get_instance(camera_device)
        except RuntimeError as e:
            QMessageBox.critical(self, "Camera Error", str(e))
            self.close()
            return

        self.layout = QVBoxLayout(self)
        
        # Preview Label
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.preview_label, 1)
        
        # Scrollable Controls Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        controls_container = QWidget()
        self.controls_layout = QVBoxLayout(controls_container)
        scroll_area.setWidget(controls_container)
        self.layout.addWidget(scroll_area)

        # Add Stream Format and Resolution Controls
        self._populate_stream_controls()

        # Populate controls based on OS
        print(f"checking os.name:{os.name}")
        # Skip native v4l2 controls for libcamera devices as they are managed differently
        if os.name == "posix" and not str(self.camera_device).startswith("libcamera:"):
            print(f"running the native v4l2 controls")
            self._populate_v4l2_controls()
        else:
            self._populate_basic_opencv_controls()
            
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def _populate_stream_controls(self):
        fmt_group = QGroupBox("Stream Format & Resolution")
        fmt_layout = QHBoxLayout(fmt_group)
        
        fmt_layout.addWidget(QLabel("Format:"))
        self.combo_fmt = QComboBox()
        fmt_layout.addWidget(self.combo_fmt)
        
        fmt_layout.addWidget(QLabel("Resolution:"))
        self.combo_res = QComboBox()
        fmt_layout.addWidget(self.combo_res)
        
        self.btn_apply_fmt = QPushButton("Apply")
        self.btn_apply_fmt.clicked.connect(self.apply_format_and_resolution)
        fmt_layout.addWidget(self.btn_apply_fmt)
        
        self.controls_layout.addWidget(fmt_group)
        
        if os.name == "posix":
            self.formats = _parse_v4l2_formats(self.camera_device)
            if self.formats:
                for fmt in self.formats:
                    self.combo_fmt.addItem(fmt["format"])
                self.combo_fmt.currentIndexChanged.connect(self.update_resolution_combo)
                if self.combo_fmt.count() > 0:
                    self.update_resolution_combo(0)
            else:
                self.combo_fmt.addItem("Default")
                self.combo_res.addItem("Default")
        else:
            self.combo_fmt.addItems(["Default", "MJPG", "YUYV"])
            self.combo_res.addItems(["Default", "640x480", "1280x720", "1920x1080"])

    def update_resolution_combo(self, index):
        self.combo_res.clear()
        if hasattr(self, 'formats') and self.formats and index < len(self.formats):
            sizes = self.formats[index]["sizes"]
            if sizes:
                self.combo_res.addItems(sizes)
            else:
                self.combo_res.addItem("Default")
        else:
            self.combo_res.addItem("Default")

    def apply_format_and_resolution(self):
        fmt_text = self.combo_fmt.currentText()
        res_text = self.combo_res.currentText()
        
        try:
            if fmt_text != "Default":
                fourcc = cv2.VideoWriter_fourcc(*fmt_text[:4])
                self.cam_mgr.set_prop(cv2.CAP_PROP_FOURCC, fourcc)
                
            if res_text != "Default":
                w, h = map(int, res_text.split('x'))
                self.cam_mgr.set_prop(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.cam_mgr.set_prop(cv2.CAP_PROP_FRAME_HEIGHT, h)
                
            QMessageBox.information(self, "Settings Applied", f"Format: {fmt_text}, Resolution: {res_text}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {e}")

    def _populate_v4l2_controls(self):
        """Heuristically enumerate and create widgets for all available V4L2 controls."""
        # Check if v4l2-ctl is available
        if not shutil.which("v4l2-ctl"):
            self._populate_basic_opencv_controls()
            return

        self.controls = _parse_v4l2_controls(self.camera_device)
        
        if not self.controls:
            self._populate_basic_opencv_controls()
            return
            
        # Group controls by prefix (e.g., "auto_exposure", "exposure_absolute" -> "Exposure Controls")
        groups = {}
        for ctrl in self.controls:
            group_name = ctrl["name"].split("_")[0].title() + " Controls"
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(ctrl)

        for group_name, ctrls in groups.items():
            grp_box = QGroupBox(group_name)
            form_layout = QFormLayout(grp_box)
            
            for ctrl in ctrls:
                if ctrl["type"] == "menu":
                    combo = QComboBox()
                    for val, text in ctrl["menu_items"]:
                        combo.addItem(text, val)
                    # Set current value
                    idx = combo.findData(ctrl["value"])
                    if idx != -1:
                        combo.setCurrentIndex(idx)
                    combo.currentIndexChanged.connect(lambda v, c=combo, n=ctrl["name"]: self._on_menu_changed(n, c.itemData(v)))
                    form_layout.addRow(ctrl["name"].replace("_", " ").title(), combo)
                    
                elif ctrl["type"] in ("int", "slider"):
                    slider = QSlider(Qt.Horizontal)
                    slider.setMinimum(ctrl["min"])
                    slider.setMaximum(ctrl["max"])
                    slider.setValue(ctrl["value"])
                    
                    spin = QSpinBox()
                    spin.setMinimum(ctrl["min"])
                    spin.setMaximum(ctrl["max"])
                    spin.setValue(ctrl["value"])
                    
                    slider.valueChanged.connect(spin.setValue)
                    spin.valueChanged.connect(slider.setValue)
                    spin.valueChanged.connect(lambda v, n=ctrl["name"]: self._on_slider_changed(n, v))
                    
                    row_layout = QHBoxLayout()
                    row_layout.addWidget(slider)
                    row_layout.addWidget(spin)
                    form_layout.addRow(ctrl["name"].replace("_", " ").title(), row_layout)
                    
                elif ctrl["type"] == "bool":
                    chk = QCheckBox()
                    chk.setChecked(ctrl["value"] == 1)
                    chk.stateChanged.connect(lambda s, n=ctrl["name"]: self._on_bool_changed(n, s))
                    form_layout.addRow(ctrl["name"].replace("_", " ").title(), chk)

            self.controls_layout.addWidget(grp_box)
            
        # Add a "Reset to Defaults" button
        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.clicked.connect(self._reset_to_defaults)
        self.controls_layout.addWidget(btn_reset)
        self.controls_layout.addStretch()

    def _populate_basic_opencv_controls(self):
        print(f"falling back to _populate_basic_opencv_controls(self)")
        """Fallback for Windows/Mac or missing v4l2-ctl."""
        grp_box = QGroupBox("Basic OpenCV Controls")
        form_layout = QFormLayout(grp_box)
        
        # Continuous Sliders
        props_slider = {
            "Brightness": cv2.CAP_PROP_BRIGHTNESS,
            "Contrast": cv2.CAP_PROP_CONTRAST,
            "Saturation": cv2.CAP_PROP_SATURATION,
            "Hue": cv2.CAP_PROP_HUE,
            "Gain": cv2.CAP_PROP_GAIN,
            "Exposure": cv2.CAP_PROP_EXPOSURE
        }
        
        self.sliders = {}
        for name, prop in props_slider.items():
            slider = QSlider(Qt.Horizontal)
            val = int(self.cam_mgr.cap.get(prop))
            slider.setValue(val)
            slider.valueChanged.connect(lambda v, p=prop: self.cam_mgr.set_prop(p, v))
            form_layout.addRow(name, slider)
            self.sliders[name] = slider
            
        # Auto/Manual Toggles
        toggle_group = QGroupBox("Auto/Manual Controls")
        t_layout = QVBoxLayout(toggle_group)
        toggles = {
            "Auto Focus": cv2.CAP_PROP_AUTOFOCUS,
            "Auto Exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
            "Auto White Balance": cv2.CAP_PROP_AUTO_WB
        }
        self.toggles = {}
        for name, prop in toggles.items():
            chk = QCheckBox(name)
            val = self.cam_mgr.cap.get(prop)
            
            if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
                is_auto = (val > 0.5)
            else:
                is_auto = (val == 1.0 or val > 0)
             
            chk.setChecked(is_auto)
            chk.stateChanged.connect(lambda s, p=prop: self.toggle_camera_prop(p, s))
            t_layout.addWidget(chk)
            self.toggles[name] = chk
            
        form_layout.addRow("", toggle_group)
        self.controls_layout.addWidget(grp_box)
        self.controls_layout.addStretch()

    def _on_slider_changed(self, name, value):
        _set_v4l2_control(self.camera_device, name, value)

    def _on_menu_changed(self, name, value):
        _set_v4l2_control(self.camera_device, name, value)

    def _on_bool_changed(self, name, state):
        # Handle both int (legacy/Qt5) and Qt.CheckState enum (PySide6/Qt6)
        is_checked = (state == Qt.CheckState.Checked) or (state == 2)
        val = 1 if is_checked else 0
        _set_v4l2_control(self.camera_device, name, val)

    def _reset_to_defaults(self):
        if hasattr(self, 'controls'):
            for ctrl in self.controls:
                if "default" in ctrl:
                    _set_v4l2_control(self.camera_device, ctrl["name"], ctrl["default"])

    def toggle_camera_prop(self, prop, state):
        """Map checkbox state to OpenCV enum (Fallback method)"""
        # Handle both int (legacy/Qt5) and Qt.CheckState enum (PySide6/Qt6)
        is_checked = (state == Qt.CheckState.Checked) or (state == 2)
        if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            val = 0.75 if is_checked else 0.25
        else:
            val = 1.0 if is_checked else 0.0
         
        self.cam_mgr.set_prop(prop, val)
     
        if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            actual = self.cam_mgr.cap.get(prop)
            if is_checked and actual < 0.5:
                self.cam_mgr.set_prop(prop, 1.0)
            elif not is_checked and actual > 0.5:
                self.cam_mgr.set_prop(prop, 0.0)

    def update_frame(self):
        ret, frame = self.cam_mgr.read_frame()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(pixmap)
            
    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


# =====================================================================
# REFINED RAG FRONT-END
# =====================================================================
class RagFrontEnd(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Vector DB FrontEnd")
        self.resize(800, 600)
        self.selected_image_path = ""
        self.selected_image_b64 = ""
        self.last_results = None
        self.active_search = None

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Search
        search_tab = QWidget()
        s_layout = QVBoxLayout(search_tab)
        
        q_box = QHBoxLayout()
        q_box.addWidget(QLabel("Query:"))
        self.inp_query = QLineEdit()
        q_box.addWidget(self.inp_query)
        s_layout.addLayout(q_box)

        i_box = QHBoxLayout()
        self.img_input_widget = ImageInputWidget()
        self.img_input_widget.image_loaded.connect(self.on_image_selected)
        self.btn_clear_img = QPushButton("Clear")
        self.btn_clear_img.clicked.connect(self.clear_image)
        i_box.addWidget(self.img_input_widget, 1)
        i_box.addWidget(self.btn_clear_img)
        s_layout.addLayout(i_box)

        # --- NEW CODE: Search Strategy & Buttons ---
        strat_box = QFormLayout()
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItems(["vector", "fts", "hybrid", "textontextvector", "textonimagevector", "imageonimagevector", "imageonvideovector", "textonvideovector"])
        strat_box.addRow("Search Strategy:", self.combo_strategy)
        s_layout.addLayout(strat_box)

        btn_box = QHBoxLayout()
        self.btn_run_text = QPushButton("Text Search")
        self.btn_run_img = QPushButton("Image Search")
        self.btn_run_multi = QPushButton("Multimodal Search")
        
        self.btn_run_text.clicked.connect(lambda: self.run_search(mode="text"))
        self.btn_run_img.clicked.connect(lambda: self.run_search(mode="image"))
        self.btn_run_multi.clicked.connect(lambda: self.run_search(mode="multi"))
        
        btn_box.addWidget(self.btn_run_text)
        btn_box.addWidget(self.btn_run_img)
        btn_box.addWidget(self.btn_run_multi)
        
        self.btn_diag = QPushButton("Open Search Results Viewer")
        self.btn_diag.setEnabled(False)
        self.btn_diag.clicked.connect(self.open_diag)
        
        s_layout.addLayout(btn_box)
        s_layout.addWidget(self.btn_diag)
        s_layout.addStretch()
        self.tabs.addTab(search_tab, "Search Node")

        # Tab 2: Config
        cfg_tab = QWidget()
        c_layout = QFormLayout(cfg_tab)
        self.cfg_ip = QLineEdit("127.0.0.1")
        self.cfg_port = QLineEdit("5001")
        self.cfg_limit = QSpinBox()
        self.cfg_limit.setRange(0, 1000)  # or .setMaximum(1000) if min is fine at 0
        self.cfg_limit.setValue(256)

        # New DB Hot-Swap Controls
        self.table_combo = QComboBox()
        self.btn_fetch_dbs = QPushButton("Fetch Available Tables") # Updated text
        self.btn_fetch_dbs.clicked.connect(self.fetch_tables)
        
        self.btn_load_db = QPushButton("Hot-Swap Table") # Updated text
        self.btn_load_db.clicked.connect(self.load_table)
        
        table_layout = QHBoxLayout()
        table_layout.addWidget(self.btn_fetch_dbs)
        table_layout.addWidget(self.table_combo, 1)
        table_layout.addWidget(self.btn_load_db)

        c_layout.addRow("RAG Server IP:", self.cfg_ip)
        c_layout.addRow("RAG Server Port:", self.cfg_port)
        c_layout.addRow("Result Limit:", self.cfg_limit)
        c_layout.addRow("Remote Table:", table_layout)
        
        config_io_layout = QHBoxLayout()
        btn_export_cfg = QPushButton("Export Config")
        btn_import_cfg = QPushButton("Import Config")
        btn_export_cfg.clicked.connect(self.export_config)
        btn_import_cfg.clicked.connect(self.import_config)
        config_io_layout.addWidget(btn_export_cfg)
        config_io_layout.addWidget(btn_import_cfg)
        c_layout.addRow("Configuration File:", config_io_layout)
        
        self.tabs.addTab(cfg_tab, "Configuration")

        # Tab 3: Image Capture
        cap_tab = QWidget()
        cap_layout = QVBoxLayout(cap_tab)
        
        # ESP32-CAM
        esp_group = QGroupBox("ESP32-CAM Capture")
        esp_layout = QHBoxLayout(esp_group)
        esp_layout.addWidget(QLabel("URL:"))
        self.esp32_url_input = QLineEdit("http://192.168.5.1/capture")
        esp_layout.addWidget(self.esp32_url_input)
        self.btn_esp32_capture = QPushButton("Capture from ESP32")
        self.btn_esp32_capture.clicked.connect(self.capture_esp32)
        esp_layout.addWidget(self.btn_esp32_capture)
        
        self.btn_esp32_settings = QPushButton("Open ESP32 Web UI")
        self.btn_esp32_settings.clicked.connect(self.open_esp32_settings)
        esp_layout.addWidget(self.btn_esp32_settings)
        cap_layout.addWidget(esp_group)
        
        # UVC Webcam
        uvc_group = QGroupBox("UVC Webcam Capture")
        uvc_layout = QHBoxLayout(uvc_group)
        uvc_layout.addWidget(QLabel("Camera:"))
        self.uvc_combo = QComboBox()
        self.populate_uvc_cameras()
        uvc_layout.addWidget(self.uvc_combo)
        self.btn_uvc_capture = QPushButton("Capture from Webcam")
        self.btn_uvc_capture.clicked.connect(self.capture_uvc)
        uvc_layout.addWidget(self.btn_uvc_capture)
        self.btn_uvc_controls = QPushButton("Camera Controls")
        self.btn_uvc_controls.clicked.connect(self.open_uvc_controls)
        uvc_layout.addWidget(self.btn_uvc_controls)
        cap_layout.addWidget(uvc_group)
        
        # Shared capture option
        opt_layout = QHBoxLayout()
        self.chk_save_capture = QCheckBox("Save captured image to disk (run directory)")
        self.chk_save_capture.setChecked(False)
        opt_layout.addWidget(self.chk_save_capture)
        opt_layout.addStretch()
        cap_layout.addLayout(opt_layout)
        
        cap_layout.addStretch()
        self.tabs.addTab(cap_tab, "Image Capture")

        self.status = QLabel("Ready")
        layout.addWidget(self.status)

#    def select_image(self):
#        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
#        if path:
#            self.selected_image_path = path
#            self.lbl_img.setText(os.path.basename(path))

    def on_image_selected(self, path, b64_data):
        self.selected_image_path = path
        self.selected_image_b64 = b64_data
        self.status.setText(f"Image selected: {os.path.basename(path)}")

    def clear_image(self):
        self.selected_image_path = ""
        self.selected_image_b64 = ""
        self.img_input_widget.clear()
        self.img_input_widget.setText("Click, Drop, or Ctrl+V to select image")
        self.status.setText("Image cleared.")

    def toggle_search_buttons(self, enabled: bool):
        self.btn_run_text.setEnabled(enabled)
        self.btn_run_img.setEnabled(enabled)
        self.btn_run_multi.setEnabled(enabled)

    def run_search(self, mode="text"):
        addr = f"tcp://{self.cfg_ip.text()}:{self.cfg_port.text()}"
        self.toggle_search_buttons(False)
        self.status.setText(f"Querying {addr} [{mode} via {self.combo_strategy.currentText()}]...")
        
        # Filter inputs based on selected button mode
        text_payload = self.inp_query.text() if mode in ("text", "multi") else ""
        img_payload_path = self.selected_image_path if mode in ("image", "multi") else ""
        img_payload_b64 = self.selected_image_b64 if mode in ("image", "multi") else ""
        
        if mode == "image" and not img_payload_b64:
            QMessageBox.warning(self, "Input Error", "Please select an image first.")
            self.toggle_search_buttons(True)
            return
            
        self.active_search = ZmqSearchThread(
            text_query=text_payload,
            image_path=img_payload_path,
            image_b64=img_payload_b64,
            limit=self.cfg_limit.value(),
            search_type=self.combo_strategy.currentText(),
            target_address=addr
        )
        self.active_search.finished_signal.connect(self.on_success)
        self.active_search.error_signal.connect(self.on_error)
        self.active_search.start()

    def on_success(self, data, req_id):
        self.toggle_search_buttons(True)
        self.last_results = data
        self.btn_diag.setEnabled(True)
        count = len(data.get('results', []))
        msg = f"Found {count} results."
        
        # Check if any result is a video frame that still needs extraction
        needs_extraction = any(
            r.get("metadata", {}).get("type") == "video_frame" and not r.get("image_path")
            for r in data.get("results", [])
        )
        if needs_extraction:
            msg += " extracting the video frames might require some time, please check the terminal for live information"
            
        self.status.setText(msg)


    def on_error(self, msg, req_id):
        self.toggle_search_buttons(True)
        self.status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "ZMQ Error", msg)

    def open_diag(self):
        if self.last_results:
            # Pass the current text query to allow auto-organizer to name the export folder
            self.diag = DiagnosticWindow(self.last_results, active_query_text=self.inp_query.text())
            self.diag.show()

    def get_address(self):
        return f"tcp://{self.cfg_ip.text()}:{self.cfg_port.text()}"

    def fetch_tables(self):
        self.toggle_search_buttons(False) # LOCK SEARCH
        self.btn_fetch_dbs.setEnabled(False)
        self.status.setText("Fetching tables...")
        
        self.cfg_thread = ZmqConfigThread({"action": "list_tables"}, self.get_address())
        self.cfg_thread.finished_signal.connect(self.on_dbs_fetched)
        self.cfg_thread.error_signal.connect(self.on_cfg_error)
        self.cfg_thread.start()

    def on_dbs_fetched(self, data):
        self.toggle_search_buttons(True) # UNLOCK SEARCH
        self.btn_fetch_dbs.setEnabled(True)
        if data.get("status") == "success":
            self.table_combo.clear()
            self.table_combo.addItems(data.get("tables", []))
            self.status.setText(f"Found {len(data.get('tables', []))} tables.")
        else:
            self.status.setText(f"Error: {data.get('message')}")

    def load_table(self):
        target_db = self.table_combo.currentText()
        if not target_db: return
        
        self.toggle_search_buttons(False) # LOCK SEARCH
        self.btn_load_db.setEnabled(False)
        self.status.setText(f"Hot-swapping to {target_db}...")
        
        self.cfg_thread = ZmqConfigThread({"action": "load_table", "table_name": target_db}, self.get_address())
        self.cfg_thread.finished_signal.connect(self.on_table_loaded)
        self.cfg_thread.error_signal.connect(self.on_cfg_error)
        self.cfg_thread.start()

    def on_table_loaded(self, data):
        self.toggle_search_buttons(True) # UNLOCK SEARCH
        self.btn_load_db.setEnabled(True)
        if data.get("status") == "success":
            self.status.setText(data.get("message", "Table loaded successfully."))
        else:
            self.status.setText(f"Failed to load Table: {data.get('message')}")

    def on_cfg_error(self, msg):
        self.toggle_search_buttons(True) # UNLOCK SEARCH
        self.btn_fetch_dbs.setEnabled(True)
        self.btn_load_db.setEnabled(True)
        self.status.setText(f"ZMQ Error: {msg}")

    def populate_uvc_cameras(self):
        self.uvc_combo.clear()
        self.uvc_combo.addItem("Scanning for cameras...", None)
        self.uvc_combo.setEnabled(False)
        
        self.cam_discovery_thread = CameraDiscoveryThread()
        self.cam_discovery_thread.finished_signal.connect(self.on_cameras_discovered)
        self.cam_discovery_thread.start()

    def on_cameras_discovered(self, devices):
        self.uvc_combo.clear()
        self.uvc_combo.setEnabled(True)
        if not devices:
            self.uvc_combo.addItem("No cameras found", None)
        else:
            for dev_path, dev_name in devices:
                self.uvc_combo.addItem(dev_name, dev_path)

    def open_esp32_settings(self):
        url = self.esp32_url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Input Error", "Please enter a URL.")
            return
        
        from urllib.parse import urlparse
        import webbrowser
        
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            QMessageBox.warning(self, "Input Error", "Invalid URL. Please include http:// or https://")
            return
            
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        webbrowser.open(base_url)
        self.status.setText(f"Opened ESP32 Web UI at {base_url}")

    def capture_esp32(self):
        url = self.esp32_url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Input Error", "Please enter a URL.")
            return
            
        self.status.setText("Capturing from ESP32-CAM...")
        QApplication.processEvents()
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            img_data = response.content
            
            # Validate image to handle partial/broken data
            nparr = np.frombuffer(img_data, np.uint8)
            img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_decode is None:
                raise ValueError("Received data is not a valid image (possibly partial or broken stream).")
                
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_hash = hashlib.md5(img_data).hexdigest()[:8]
            source_name = f"esp32_{ts}_{frame_hash}.jpg"
            
            if self.img_input_widget.load_image_from_bytes(img_data, source_name):
                if self.chk_save_capture.isChecked():
                    save_path = os.path.join(os.getcwd(), f"capture_{source_name}")
                    with open(save_path, "wb") as f:
                        f.write(img_data)
                    self.status.setText(f"Saved to {save_path}")
                else:
                    self.status.setText("ESP32-CAM image captured to memory.")
            else:
                raise ValueError("Failed to load image into preview widget.")
                
        except requests.exceptions.RequestException as e:
            self.status.setText(f"Network Error: {str(e)}")
            QMessageBox.critical(self, "Capture Error", f"Network error: {str(e)}")
        except ValueError as e:
            self.status.setText(f"Image Error: {str(e)}")
            QMessageBox.critical(self, "Capture Error", str(e))
        except Exception as e:
            self.status.setText(f"Error: {str(e)}")
            QMessageBox.critical(self, "Capture Error", str(e))

    def capture_uvc(self):
        cam_device = self.uvc_combo.currentData()
        if cam_device is None:
            QMessageBox.warning(self, "Input Error", "No camera selected.")
            return
            
        self.status.setText("Capturing from UVC Webcam...")
        QApplication.processEvents()
        
        try:
            cam_mgr = CameraManager.get_instance(cam_device)
            ret, frame = cam_mgr.read_frame()
            
            if ret:
                _, buffer = cv2.imencode(".png", frame)
                img_data = buffer.tobytes()
                
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_hash = hashlib.md5(img_data).hexdigest()[:8]
                source_name = f"uvc_{ts}_{frame_hash}.png"
                
                if self.img_input_widget.load_image_from_bytes(img_data, source_name):
                    if self.chk_save_capture.isChecked():
                        save_path = os.path.join(os.getcwd(), f"capture_{source_name}")
                        with open(save_path, "wb") as f:
                            f.write(img_data)
                        self.status.setText(f"Saved to {save_path}")
                    else:
                        self.status.setText(f"Frame captured to memory ({ts}_{frame_hash})")
                else:
                    raise ValueError("Failed to load frame into preview widget.")
            else:
                QMessageBox.critical(self, "Capture Error", "Failed to read frame.")
        except Exception as e:
            self.status.setText(f"Error: {str(e)}")
            QMessageBox.critical(self, "Capture Error", str(e))


    def open_uvc_controls(self):
        cam_device = self.uvc_combo.currentData()
        if cam_device is None:
            QMessageBox.warning(self, "Input Error", "No camera selected.")
            return
        self.uvc_controls_window = UvcCameraControlsWindow(cam_device)
        self.uvc_controls_window.show()


    def export_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Configuration", "", "JSON Files (*.json)")
        if not path: return
        
        config_data = {
            "rag_server_ip": self.cfg_ip.text(),
            "rag_server_port": self.cfg_port.text(),
            "result_limit": self.cfg_limit.value(),
            "esp32_url": self.esp32_url_input.text()
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(config_data, f, indent=4)
            self.status.setText(f"Configuration exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def import_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Configuration", "", "JSON Files (*.json)")
        if not path: return
        
        try:
            with open(path, 'r') as f:
                config_data = json.load(f)
                
            self.cfg_ip.setText(config_data.get("rag_server_ip", "127.0.0.1"))
            self.cfg_port.setText(config_data.get("rag_server_port", "5001"))
            self.cfg_limit.setValue(config_data.get("result_limit", 256))
            self.esp32_url_input.setText(config_data.get("esp32_url", "http://192.168.5.1/capture"))
            
            self.status.setText(f"Configuration imported from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RagFrontEnd()
    window.show()
    sys.exit(app.exec())
