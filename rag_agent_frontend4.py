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


from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSpinBox, QTextEdit, 
    QTabWidget, QListWidget, QGraphicsView, QGraphicsScene, 
    QSplitter, QMessageBox, QGroupBox, QFormLayout, QLineEdit,
    QComboBox, QCheckBox, QListWidgetItem
)
from PySide6.QtCore import Qt, QThread, Signal, QRectF, QSize
from PySide6.QtGui import QPixmap, QImage, QWheelEvent, QMouseEvent, QGuiApplication


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

    def __init__(self, text_query, image_path, limit, search_type, target_address, request_id="local_gui", timeout_ms=60000):
        super().__init__()
        self.text_query = text_query
        self.image_path = image_path
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

            img_b64 = ""
            if self.image_path and os.path.exists(self.image_path):
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
    image_loaded = Signal(str)

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
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        pixmap.save(tmp.name, "PNG")
                        self.load_image_from_path(tmp.name)
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
            self.image_loaded.emit(path)
        else:
            self.setText("Failed to load image.")

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
class RagFrontEnd(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Vector DB Manager")
        self.resize(800, 600)
        self.selected_image_path = ""
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
        
        self.btn_diag = QPushButton("Open Diagnostic Viewer")
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
        
        self.tabs.addTab(cfg_tab, "Configuration")



        self.status = QLabel("Ready")
        layout.addWidget(self.status)

#    def select_image(self):
#        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
#        if path:
#            self.selected_image_path = path
#            self.lbl_img.setText(os.path.basename(path))

    def on_image_selected(self, path):
        self.selected_image_path = path
        self.status.setText(f"Image selected: {os.path.basename(path)}")

    def clear_image(self):
        self.selected_image_path = ""
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
        img_payload = self.selected_image_path if mode in ("image", "multi") else ""
        
        if mode == "image" and not img_payload:
            QMessageBox.warning(self, "Input Error", "Please select an image first.")
            self.toggle_search_buttons(True)
            return
            
        self.active_search = ZmqSearchThread(
            text_query=text_payload,
            image_path=img_payload,
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RagFrontEnd()
    window.show()
    sys.exit(app.exec())