import sys
import os
import json
import uuid
import hashlib
import numpy as np
import pyarrow as pa
import lancedb
import psutil
import gc

# --- FIX: tell PyArrow's allocator to return unused memory to the OS ---
# Without this, the allocator keeps freed pages in its own pool forever,
# which is what makes RSS/swap appear to monotonically grow.
try:
    # mimalloc backend (default on most builds)
    pa.mimalloc_set_decay_ms(10)
except Exception:
    pass
try:
    # jemalloc backend (some builds)
    pa.jemalloc_set_decay_ms(10)
except Exception:
    pass
from lancedb.rerankers import RRFReranker
import zmq
import warnings
import logging
import subprocess
import tempfile
import re


from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QGroupBox,
    QSplitter, QMessageBox, QFormLayout, QComboBox, QProgressBar,
    QTabWidget, QCheckBox, QDoubleSpinBox
)

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont

# =====================================================================
# ASYNC MODEL DOWNLOADER / LOADER
# =====================================================================
class ModelLoaderThread(QThread):
    """
    Downloads and loads HuggingFace models in the background so the GUI 
    doesn't freeze. Relies on local cache if already downloaded.
    """
    log_signal = Signal(str)
    finished_signal = Signal(object, str) # model_obj, error_msg

    def __init__(self, model_cfg: dict):
        super().__init__()
        self.model_cfg = model_cfg

    def run(self):
        # Suppress HuggingFace and PyTorch deprecation warnings/logs from polluting the console
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"  
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        m_type = self.model_cfg.get("type")
        repo_id = self.model_cfg.get("repo_id")
        
        self.log_signal.emit(f"[MODEL] Initializing load/download for: {repo_id}")
        
        try:
            if m_type == "dummy":
                self.finished_signal.emit("dummy", "")
                return

            import torch
            
            # --- HARDWARE DETECTION ---
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_signal.emit(f"[MODEL] Hardware Target: {device.upper()}")
            # -------------------------------

            if m_type == "sentence-transformers":
                from huggingface_hub import snapshot_download
                from sentence_transformers import SentenceTransformer
                
                self.log_signal.emit("[MODEL] Synchronizing repository scripts...")
                snapshot_download(
                    repo_id=repo_id,
                    ignore_patterns=["*.bin", "*.onnx", "*.h5", "*.msgpack", "*.ot", "*.ckpt"]
                )
                
                # Standard CLIP loads safely and natively without dtype hacks
                model = SentenceTransformer(repo_id, trust_remote_code=True, device=device)
                self.finished_signal.emit(model, "")
                
            else:
                self.finished_signal.emit(None, f"Unknown model type: {m_type}")
                
        except ImportError as e:
            err = f"Missing dependency. Please run: pip install torch torchvision sentence-transformers transformers pillow\nDetails: {e}"
            self.finished_signal.emit(None, err)
        except Exception as e:
            self.finished_signal.emit(None, str(e))

# =====================================================================
# LIVE DIRECTORY MONITOR THREAD
# =====================================================================
class LiveMonitorThread(QThread):
    files_found_signal = Signal(list)

    def __init__(self, directory, recursive, exts):
        super().__init__()
        self.directory = directory
        self.recursive = recursive
        self.exts = exts
        self.running = True
        self.known_files = set()

    def run(self):
        while self.running:
            found = []
            if os.path.exists(self.directory):
                for root, dirs, files in os.walk(self.directory):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in self.exts:
                            full_path = os.path.join(root, file)
                            if full_path not in self.known_files:
                                found.append(full_path)
                                self.known_files.add(full_path)
                    if not self.recursive:
                        break # Stop after top level

            if found:
                self.files_found_signal.emit(found)

            # Sleep for 10 seconds before next scan
            for _ in range(10):
                if not self.running: break
                self.msleep(1000)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

# =====================================================================
# NETWORK THREAD (ZeroMQ TCP Server)
# =====================================================================
class ZmqServerThread(QThread):
    log_signal = Signal(str)
    request_signal = Signal(dict)

    def __init__(self, bind_ip, port):
        super().__init__()
        self.bind_ip = bind_ip
        self.port = port
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.response_data = None

    def run(self):
        bind_addr = f"tcp://{self.bind_ip}:{self.port}"
        try:
            self.socket.bind(bind_addr)
            self.log_signal.emit(f"[NETWORK] Listening on {bind_addr}")
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Could not bind to {bind_addr}: {e}")
            return

        while self.running:
            try:
                if self.socket.poll(100):
                    message = self.socket.recv_json()
                    self.log_signal.emit(f"[NETWORK] Received Request: {json.dumps(message)[:512]}...")
                    
                    self.request_signal.emit(message)
                    
                    while self.response_data is None and self.running:
                        self.msleep(10)
                        
                    if self.running:
                        self.socket.send_json(self.response_data)
                        self.log_signal.emit("[NETWORK] Reply sent.")
                        self.response_data = None

            except Exception as e:
                if self.running:
                    self.log_signal.emit(f"[NETWORK ERROR] {e}")

    def send_response(self, data: dict):
        self.response_data = data

    def stop(self):
        self.running = False
        self.socket.close()
        self.context.term()
        self.quit()
        self.wait()

# =====================================================================
# MAIN GUI & SYSTEM
# =====================================================================
class RagAgentWindow(QMainWindow):

    def _log_memory(self, label: str):
        try:
            #import psutil, gc
            gc.collect()
            proc = psutil.Process(os.getpid())
            rss_mb  = proc.memory_info().rss  / 1024 / 1024
            vms_mb  = proc.memory_info().vms  / 1024 / 1024
            self.log_diag(f"[MEM] {label:40s}  RSS={rss_mb:8.1f} MB  VMS={vms_mb:8.1f} MB")
        except Exception:
            pass

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimodal RAG Server")
        self.resize(1100, 800)

        # Agent State
        self.db = None
        self.table = None
        self.server_thread = None
        self.monitor_thread = None
        self.processed_files = set()
        
        # Buffer State for Decoupled Ingestion
        self.ingest_buffer = []
        self.ingest_buffer_key = ""
        
        # Model State
        self.active_model_obj = "dummy"
        self.is_ingesting = False
        self.stop_requested = False
        self.model_loader_thread = None
        
        # Simplified Model Registry
        self.models_registry = [
            {"name": "Dummy (Instant Start/Debug)", "repo_id": "dummy", "type": "dummy", "dimension": 512},
            # --- THE STABLE SOLUTION ---
            {"name": "CLIP ViT-B-32 (Stable OpenCLIP)", "repo_id": "sentence-transformers/clip-ViT-B-32", "type": "sentence-transformers", "dimension": 512},
            {"name": "CLIP ViT-B-32 multilingual-v1 (Recommended for Japanese searches on metadata / images)", "repo_id": "sentence-transformers/clip-ViT-B-32-multilingual-v1", "type": "sentence-transformers", "dimension": 512},
            {"name": "CLIP ViT-L-14 (Bigger model, much higher accuracy, requires entirely separate tables due to different vector sizes)", "repo_id": "sentence-transformers/clip-ViT-L-14", "type": "sentence-transformers", "dimension": 768},
        ]

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # ==========================================
        # TOP CONFIGURATION PANEL
        # ==========================================
        config_group = QGroupBox("Master Configuration")
        config_layout = QHBoxLayout()

        # Left: Server & DB
        net_layout = QFormLayout()
        self.ip_input = QLineEdit("127.0.0.1")
        self.port_input = QLineEdit("5001")
        
        dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit(os.path.join(os.getcwd(), "lancedb_data"))
        self.dir_btn = QPushButton("Browse...")
        self.dir_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(self.dir_btn)

        db_layout = QHBoxLayout()
        self.db_combo = QComboBox()
        self.db_combo.currentIndexChanged.connect(self.on_db_selected)
        
        self.scan_db_btn = QPushButton("Scan")
        self.scan_db_btn.setFixedWidth(60)
        self.scan_db_btn.clicked.connect(self.scan_for_databases)
        
        db_layout.addWidget(self.db_combo)
        db_layout.addWidget(self.scan_db_btn)

        # Table Creation Layout
        table_create_layout = QHBoxLayout()
        self.table_input = QLineEdit("general_collection")
        self.create_table_btn = QPushButton("Create New")
        self.create_table_btn.clicked.connect(self.create_new_table)
        table_create_layout.addWidget(self.table_input)
        table_create_layout.addWidget(self.create_table_btn)

        # Table Selection Layout
        table_select_layout = QHBoxLayout()
        self.table_combo = QComboBox()
        self.scan_tables_btn = QPushButton("Scan Tables")
        self.scan_tables_btn.clicked.connect(self.scan_for_tables)
        self.load_table_btn = QPushButton("Load Selected")
        self.load_table_btn.clicked.connect(self.load_selected_table)
        table_select_layout.addWidget(self.table_combo)
        table_select_layout.addWidget(self.scan_tables_btn)
        table_select_layout.addWidget(self.load_table_btn)

        net_layout.addRow("Bind IP:", self.ip_input)
        net_layout.addRow("TCP Port:", self.port_input)
        net_layout.addRow("Work Directory:", dir_layout)
        net_layout.addRow("Available DBs:", db_layout)
        net_layout.addRow("New Table Name:", table_create_layout)
        net_layout.addRow("Available Tables:", table_select_layout)

        # Middle: Model Manager
        model_layout = QFormLayout()
        
        self.model_combo = QComboBox()
        self.refresh_model_dropdown()
        
        self.load_model_btn = QPushButton("Load / Download Selected Model")
        self.load_model_btn.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        self.load_model_btn.clicked.connect(self.trigger_model_load)
        
        model_layout.addRow("Select Model:", self.model_combo)
        model_layout.addRow("", self.load_model_btn)

        # Right: Actions
        btn_layout = QVBoxLayout()
        self.save_cfg_btn = QPushButton("Save Config")
        self.load_cfg_btn = QPushButton("Load Config")
        self.start_srv_btn = QPushButton("Start TCP Server")
        self.stop_ingest_btn = QPushButton("Stop Ingestion")
        self.stop_ingest_btn.setStyleSheet("background-color: #a33; color: white; font-weight: bold;")
        self.stop_ingest_btn.clicked.connect(self.stop_ongoing_ingestion)
        self.stop_ingest_btn.setEnabled(False)
        
        self.save_cfg_btn.clicked.connect(self.save_config)
        self.load_cfg_btn.clicked.connect(self.load_config)
        self.start_srv_btn.clicked.connect(self.toggle_server)

        self.chk_return_img_b64 = QCheckBox("Return Local Images as Base64 in Search")
        self.chk_return_img_b64.setChecked(False)

        btn_layout.addWidget(self.save_cfg_btn)
        btn_layout.addWidget(self.load_cfg_btn)
        btn_layout.addWidget(self.start_srv_btn)
        btn_layout.addWidget(self.stop_ingest_btn)
        btn_layout.addWidget(self.chk_return_img_b64)
        btn_layout.addStretch()

        config_layout.addLayout(net_layout, stretch=2)
        config_layout.addLayout(model_layout, stretch=2)
        config_layout.addLayout(btn_layout, stretch=1)
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        # ==========================================
        # MIDDLE SPLITTER (Ingestion vs Logs)
        # ==========================================
        splitter = QSplitter(Qt.Horizontal)

        # --- LEFT SIDE ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.ingest_tabs = QTabWidget()

        # TAB 1: Single Ingestion
        tab_single = QWidget()
        ingest_layout = QFormLayout(tab_single)
        
        self.ingest_text = QLineEdit()
        self.ingest_text.setPlaceholderText("Description or extracted text...")
        
        img_layout = QHBoxLayout()
        self.ingest_img = QLineEdit()
        self.ingest_img.setPlaceholderText("Path to PNG/TIFF/JPG...")
        self.ingest_img_btn = QPushButton("...")
        self.ingest_img_btn.setFixedWidth(30)
        self.ingest_img_btn.clicked.connect(self.browse_image_ingest)
        img_layout.addWidget(self.ingest_img)
        img_layout.addWidget(self.ingest_img_btn)

        self.ingest_meta = QLineEdit('{"source": "manual", "tags": []}')

        self.ingest_btn = QPushButton("Embed and Add to LanceDB")
        self.ingest_btn.clicked.connect(self.ingest_data)

        ingest_layout.addRow("Text:", self.ingest_text)
        ingest_layout.addRow("Image:", img_layout)
        ingest_layout.addRow("Metadata (JSON):", self.ingest_meta)
        ingest_layout.addRow("", self.ingest_btn)

        # TAB 2: Batch & Auto Monitoring
        tab_batch = QWidget()
        batch_layout = QFormLayout(tab_batch)
        
        bdir_layout = QHBoxLayout()
        self.batch_dir_input = QLineEdit()
        self.batch_dir_input.setPlaceholderText("Select folder with images/PDFs...")
        self.batch_dir_btn = QPushButton("Browse...")
        self.batch_dir_btn.clicked.connect(self.browse_batch_directory)
        bdir_layout.addWidget(self.batch_dir_input)
        bdir_layout.addWidget(self.batch_dir_btn)

        self.chk_assoc = QCheckBox("Auto-associate matching .txt and .json files")
        self.chk_assoc.setChecked(True)
        self.chk_recurse = QCheckBox("Recursive (Scan Subfolders)")
        self.chk_pdf = QCheckBox("Process PDFs (Requires PyMuPDF)")
        self.chk_csv = QCheckBox("Scan and ingest CSV files")
        self.chk_archive = QCheckBox("Process Compressed Archives (ZIP/TAR/7Z)")
        self.chk_csv_meta = QCheckBox("Use first CSV row as metadata fields")
        self.chk_csv_meta.setChecked(True)
        
        from PySide6.QtWidgets import QSpinBox
        self.csv_window_size = QSpinBox()
        self.csv_window_size.setRange(1, 50000)
        self.csv_window_size.setValue(256)
        
        # NEW: Global configurable buffer limit for batch/video ingestion
        self.batch_buffer_size = QSpinBox()
        self.batch_buffer_size.setRange(1, 1048576)
        self.batch_buffer_size.setValue(10000)

        # NEW: Embedding Batch Size Controls
        self.text_batch_size = QSpinBox()
        self.text_batch_size.setRange(1, 65536)
        self.text_batch_size.setValue(64)

        self.image_batch_size = QSpinBox()
        self.image_batch_size.setRange(1, 4096)
        self.image_batch_size.setValue(64)

        self.pdf_batch_size = QSpinBox()
        self.pdf_batch_size.setRange(1, 4096)
        self.pdf_batch_size.setValue(32)

        # NEW: Video Embedding Batch Size Control

        self.video_batch_size = QSpinBox()
        self.video_batch_size.setRange(1, 4096)
        self.video_batch_size.setValue(32)

        # NEW: Archive Embedding Batch Size Control
        self.archive_batch_size = QSpinBox()
        self.archive_batch_size.setRange(1, 4096)
        self.archive_batch_size.setValue(32)

        self.chk_monitor = QCheckBox("Enable Live Periodic Monitoring")

        
        self.chk_monitor.toggled.connect(self.toggle_live_monitor)

        self.batch_btn = QPushButton("Run Manual Batch Ingest Now")
        self.batch_btn.clicked.connect(self.start_batch_ingest)

        batch_layout.addRow("Source Folder:", bdir_layout)
        batch_layout.addRow("", self.chk_assoc)
        batch_layout.addRow("", self.chk_recurse)
        batch_layout.addRow("", self.chk_pdf)
        batch_layout.addRow("", self.chk_csv)
        batch_layout.addRow("", self.chk_csv_meta)
        batch_layout.addRow("CSV Batch Window:", self.csv_window_size)
        batch_layout.addRow("Record Buffer Limit:", self.batch_buffer_size)
        batch_layout.addRow("Text Embedding Batch:", self.text_batch_size)
        batch_layout.addRow("Image Embedding Batch:", self.image_batch_size)
        batch_layout.addRow("PDF Page Embedding Batch:", self.pdf_batch_size)
        batch_layout.addRow("Video Frame Embedding Batch:", self.video_batch_size) # NEW
        batch_layout.addRow("", self.chk_archive)
        batch_layout.addRow("Archive Image Embedding Batch:", self.archive_batch_size) # NEW
        batch_layout.addRow("", self.chk_monitor)
        batch_layout.addRow("", self.batch_btn)

        self.ingest_tabs.addTab(tab_single, "Single Item")
        self.ingest_tabs.addTab(tab_batch, "Batch / Auto")
        
        # TAB 3: Video Pipeline
        tab_video = QWidget()
        video_layout = QFormLayout(tab_video)

        vdir_layout = QHBoxLayout()
        self.video_dir_input = QLineEdit()
        self.video_dir_input.setPlaceholderText("Select folder with videos...")
        self.video_dir_btn = QPushButton("Browse...")
        self.video_dir_btn.clicked.connect(self.browse_video_directory)
        vdir_layout.addWidget(self.video_dir_input)
        vdir_layout.addWidget(self.video_dir_btn)

        self.video_fps = QDoubleSpinBox()
        self.video_fps.setRange(0.1, 30.0)
        self.video_fps.setSingleStep(0.5)
        self.video_fps.setValue(1.0)

        self.chk_smart_video_skip = QCheckBox("Smart Video Sampling Skip (Check last 4 frames)")
        self.chk_smart_video_skip.setChecked(False)

        self.video_progress = QProgressBar()
        self.video_progress.setValue(0)

        self.video_ingest_btn = QPushButton("Extract Frames & Embed to LanceDB")
        self.video_ingest_btn.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        self.video_ingest_btn.clicked.connect(self.run_video_ingestion)

        video_layout.addRow("Source Folder:", vdir_layout)
        video_layout.addRow("Extraction FPS:", self.video_fps)
        video_layout.addRow(" ", self.chk_smart_video_skip)
        video_layout.addRow(" ", self.video_progress)
        video_layout.addRow("", self.video_ingest_btn)

        self.ingest_tabs.addTab(tab_video, "Video Pipeline")

        
        left_layout.addWidget(self.ingest_tabs)

        query_group = QGroupBox("Manual Debug Query")
        query_layout = QFormLayout()
        
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Semantic Search (e.g. 'A red car')...")
        
        
        
        from PySide6.QtWidgets import QSpinBox
        self.query_limit = QSpinBox()
        self.query_limit.setRange(1, 256) #limit to max 500 (LanceDB hard limit for WHERE conditions), using more can and will cause segfaults in the module
        self.query_limit.setValue(5) # Default to 5 instead of 3

        # --- NEW CODE: Strategy Dropdown & FTS Index Builder ---
        self.query_strategy = QComboBox()
        self.query_strategy.addItems(["vector", "fts", "hybrid"])
        
        self.build_fts_btn = QPushButton("Rebuild FTS Index (Required for FTS/Hybrid)")
        self.build_fts_btn.setStyleSheet("background-color: #555555; color: white;")
        self.build_fts_btn.clicked.connect(self.build_fts_index)
        
        self.build_vec_btn = QPushButton("Build Vector Index (Post-Ingestion)")
        self.build_vec_btn.setStyleSheet("background-color: #555555; color: white;")
        self.build_vec_btn.clicked.connect(self.build_vector_index)

        
        self.query_btn = QPushButton("Search Database")
        self.query_btn.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        self.query_btn.clicked.connect(self.manual_query)
        # -------------------------------------------------------
        
        query_layout.addRow("Vector Prompt:", self.query_input)
        query_layout.addRow("Search Strategy:", self.query_strategy) # NEW
        query_layout.addRow("Max Results:", self.query_limit)
        query_layout.addRow("", self.build_fts_btn) # NEW
        query_layout.addRow("", self.build_vec_btn) # NEW
        query_layout.addRow("", self.query_btn)

        
        query_group.setLayout(query_layout)

        left_layout.addWidget(query_group)
        left_layout.addStretch()

        # --- RIGHT SIDE ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.diag_log = QTextEdit()
        self.diag_log.setReadOnly(True)
        self.diag_log.setFont(QFont("Consolas", 9))
        
        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        self.output_log.setFont(QFont("Consolas", 10))

        diag_group = QGroupBox("Diagnostic Log")
        dl = QVBoxLayout(); dl.addWidget(self.diag_log); diag_group.setLayout(dl)
        
        out_group = QGroupBox("Live Output Data")
        ol = QVBoxLayout(); ol.addWidget(self.output_log)
        
        copy_btn = QPushButton("Copy Output to Clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.output_log.toPlainText()))
        ol.addWidget(copy_btn)
        out_group.setLayout(ol)

        right_layout.addWidget(diag_group)
        right_layout.addWidget(out_group)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([450, 650])

        main_layout.addWidget(splitter)

        self.log_diag("System Initialized. Default 'Dummy' model active.")

    # =====================================================================
    # UI LOGIC & CONFIG
    # =====================================================================
    def log_diag(self, msg: str):
        self.diag_log.append(msg)
        # Prevent unbounded memory growth in the diagnostic log widget during long sessions
        if self.diag_log.document().blockCount() > 10000:
            current_text = self.diag_log.toPlainText()
            lines = current_text.split('\n')
            self.diag_log.setPlainText('\n'.join(lines[-5000:]))


    def refresh_model_dropdown(self):
        self.model_combo.clear()
        for m in self.models_registry:
            self.model_combo.addItem(f"{m['name']} ({m['repo_id']})", m)

    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select LanceDB Work Directory")
        if dir_path:
            self.dir_input.setText(dir_path)

    # --- NEW CODE: Recursive Database Scanner ---
    def scan_for_databases(self):
        base_dir = self.dir_input.text()
        self.log_diag(f"[DB] Recursively scanning '{base_dir}' for databases...")
        found_dbs = set()
        
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                # LanceDB tables are directories that end with .lance
                for d in dirs:
                    if d.endswith('.lance'):
                        found_dbs.add(root)

        self.db_combo.blockSignals(True)
        self.db_combo.clear()
        
        if found_dbs:
            for db_path in sorted(list(found_dbs)):
                # Display the path in the combo box and store the path in itemData
                self.db_combo.addItem(db_path, db_path)
            self.log_diag(f"[DB] Found {len(found_dbs)} database locations.")
        else:
            self.db_combo.addItem("No LanceDB tables found", None)
            self.log_diag("[DB] No LanceDB tables found in the specified directory.")
            
        self.db_combo.blockSignals(False)

    def on_db_selected(self, index):
        if index >= 0:
            selected_db = self.db_combo.itemData(index)
            if selected_db:
                self.dir_input.setText(selected_db)
                self.connect_db()
    # ------------------------------------------


    def browse_image_ingest(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.tiff *.jpg)")
        if file_path:
            self.ingest_img.setText(file_path)

    def browse_batch_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Source Directory")
        if dir_path:
            self.batch_dir_input.setText(dir_path)

    def save_config(self):
        cfg = {
            "ip": self.ip_input.text(),
            "port": self.port_input.text(),
            "work_dir": self.dir_input.text(),
            "table_name": self.table_input.text(),
            "active_model_index": self.model_combo.currentIndex(),
            "batch_buffer_size": self.batch_buffer_size.value(),
            "text_batch_size": self.text_batch_size.value(),
            "image_batch_size": self.image_batch_size.value(),
            "pdf_batch_size": self.pdf_batch_size.value(),
            "video_batch_size": self.video_batch_size.value(), # NEW
            "archive_batch_size": self.archive_batch_size.value() # NEW
        }

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(cfg, f, indent=4)
            self.log_diag(f"Config saved to {file_path}")

    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    cfg = json.load(f)
                
                self.ip_input.setText(cfg.get("ip", "127.0.0.1"))
                self.port_input.setText(cfg.get("port", "5001"))
                self.dir_input.setText(cfg.get("work_dir", ""))
                self.table_input.setText(cfg.get("table_name", "general_collection"))
                
                idx = cfg.get("active_model_index", 0)
                if 0 <= idx < self.model_combo.count():
                    self.model_combo.setCurrentIndex(idx)
                    
                self.batch_buffer_size.setValue(cfg.get("batch_buffer_size", 5000))
                self.text_batch_size.setValue(cfg.get("text_batch_size", 64))
                self.image_batch_size.setValue(cfg.get("image_batch_size", 64))
                self.pdf_batch_size.setValue(cfg.get("pdf_batch_size", 32))
                self.video_batch_size.setValue(cfg.get("video_batch_size", 32)) # NEW
                self.archive_batch_size.setValue(cfg.get("archive_batch_size", 32)) # NEW
                    
                self.log_diag(f"Config loaded from {file_path}")


            except Exception as e:
                self.log_diag(f"[CONFIG ERROR] Failed to load config: {e}")

    # =====================================================================
    # MODEL LOADING LOGIC
    # =====================================================================
    def trigger_model_load(self):
        model_cfg = self.model_combo.currentData()
        
        self.load_model_btn.setEnabled(False)
        self.load_model_btn.setText("Loading... (Check Diagnostics)")
        
        self.model_loader_thread = ModelLoaderThread(model_cfg)
        self.model_loader_thread.log_signal.connect(self.log_diag)
        self.model_loader_thread.finished_signal.connect(self.on_model_loaded)
        self.model_loader_thread.start()

    @Slot(object, str)
    def on_model_loaded(self, model_obj, error_msg):
        self.load_model_btn.setEnabled(True)
        self.load_model_btn.setText("Load / Download Selected Model")
        
        if error_msg:
            self.log_diag(f"[MODEL ERROR] {error_msg}")
            QMessageBox.critical(self, "Model Load Failed", error_msg)
            return

        self.active_model_obj = model_obj
        
        active_cfg = self.model_combo.currentData()
        self.log_diag(f"[MODEL] Successfully loaded: {active_cfg['name']}")
        
        # When model changes, we must reconnect to DB so schema dimension matches!
        self.connect_db()

    def do_embedding(self, text_query: str, image_path: str = "", image_base64: str = "") -> list[float]:
        """ Routes the embedding request based on the loaded model type """
        active_cfg = self.model_combo.currentData()
        m_type = active_cfg.get("type")
        dim = active_cfg.get("dimension")
        
        if m_type == "dummy" or self.active_model_obj == "dummy":
            vec = np.random.rand(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            return vec.tolist()

        import torch
        from PIL import Image
        import base64
        import io

        try:
            with torch.no_grad():
                # Force evaluation mode
                if hasattr(self.active_model_obj, "eval"):
                    self.active_model_obj.eval()

                # --- NATIVE INFERENCE ---
                if m_type == "sentence-transformers":
                    has_img = bool(image_base64) or bool(image_path and os.path.exists(image_path))
                    has_txt = bool(text_query and text_query.strip())
                    
                    if has_img:
                        if image_base64:
                            image_data = base64.b64decode(image_base64)
                            img = Image.open(io.BytesIO(image_data)).convert("RGB")
                        else:
                            # Fallback for manual ingestion/searches inside the RAG GUI
                            img = Image.open(image_path).convert("RGB")

                        if has_txt:
                            self.log_diag("[MODEL] Mixed query detected. Averaging vectors.")
                            img_feat = self.active_model_obj.encode(img)
                            txt_feat = self.active_model_obj.encode(text_query)
                            features = (img_feat + txt_feat) / 2.0
                        else:
                            features = self.active_model_obj.encode(img)
                    else:
                        features = self.active_model_obj.encode(text_query)
                        
                # Handle output formatting safely
                if hasattr(features, "detach"):
                    features = features.detach().cpu().numpy()
                    
                # Robust tensor conversion and flattening
                features = np.array(features, dtype=np.float32)
                if features.ndim == 2:
                    features = features[0]
                    
                # LIVE LOGGING
                raw_snippet = [round(float(x), 4) if not np.isnan(x) else "NaN" for x in features[:5]]
                self.log_diag(f"[DEBUG] Raw vector shape: {features.shape} | First 5: {raw_snippet}")
                
                nan_count = np.isnan(features).sum()
                inf_count = np.isinf(features).sum()
                if nan_count > 0 or inf_count > 0:
                    self.log_diag(f"[CRITICAL] Model output contained {nan_count} NaNs and {inf_count} Infs!")
                
                # Normalize vector to ensure uniform distance math in LanceDB
                norm = np.linalg.norm(features)
                if norm > 0 and not np.isnan(norm):
                    features = features / norm
                    
                return features.tolist()
                        
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")

    # =====================================================================
    # LANCEDB LOGIC
    # =====================================================================
    def connect_db(self, specific_table=None, force_create=False):
        work_dir = self.dir_input.text()
        active_dim = self.model_combo.currentData().get("dimension", 512)

        try:
            self.db = lancedb.connect(work_dir)
            self.log_diag(f"[DB] Connected to LanceDB at {work_dir}")
            
            # Fetch existing tables right away to handle default logic
            existing_tables = self.db.list_tables().tables
            
            # Sync combo box seamlessly
            self.table_combo.blockSignals(True)
            self.table_combo.clear()
            self.table_combo.addItems(existing_tables)
            self.table_combo.blockSignals(False)

            # Determine which table name to target
            if specific_table:
                target_table = specific_table
            else:
                target_table = self.table_input.text().strip() or "general_collection"

            # FIX: If we aren't forcing creation, and the target table doesn't exist,
            # but OTHER tables DO exist, auto-select the first available one instead of creating blindly.
            if not force_create and target_table not in existing_tables and len(existing_tables) > 0:
                target_table = existing_tables[0]
                self.log_diag(f"[DB] Auto-selected existing table '{target_table}' instead of creating '{self.table_input.text()}'")

            # Update GUI to reflect the active choice
            self.table_input.setText(target_table)
            self.table_combo.setCurrentText(target_table)

            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("text_vector", pa.list_(pa.float32(), active_dim)),
                pa.field("image_vector", pa.list_(pa.float32(), active_dim)),
                pa.field("video_vector", pa.list_(pa.float32(), active_dim)),
                pa.field("text", pa.string()),
                pa.field("image_path", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("text_hash", pa.string()),
                pa.field("image_hash", pa.string()),
                pa.field("metadata_hash", pa.string()),
                pa.field("empty_text", pa.bool_()),
                pa.field("empty_image", pa.bool_()),
                pa.field("empty_video", pa.bool_()),
                pa.field("type", pa.string()),
                pa.field("fps", pa.float32()),
                pa.field("content_key", pa.string())
            ])



            if target_table in existing_tables:
                self.table = self.db.open_table(target_table)
                # Retrieve the list of all constructed indices
                indices = self.table.list_indices()
                for tindex in indices:
                    print(f"Index Name: {tindex.name}")
                    print(f"Type: {tindex.index_type}")
                    print(f"Columns: {tindex.columns}")
    
                    # If using newer LanceDB core versions, extended metadata can be accessed:
                    if hasattr(tindex, "index_details"):
                        print(f"Details: {tindex.index_details}")
                        
                # Runtime check for deduplication fields
                schema_fields = {f.name for f in self.table.schema}
                required_fields = {"text_hash", "image_hash", "metadata_hash"}
                if not required_fields.issubset(schema_fields):
                    raise ValueError(f"Table '{target_table}' missing deduplication fields. Expected: {required_fields}")
                self.log_diag(f"[DB] Opened '{target_table}'. Total records: {len(self.table)}")

            else:
                self.table = self.db.create_table(target_table, schema=schema)
                self.log_diag(f"[DB] Created new table '{target_table}' with vector dimension {active_dim}.")
                
                # Create BTREE scalar indices on upsert keys to accelerate merge_insert joins
                self.table.create_scalar_index("content_key", index_type="BTREE")
                self.table.create_scalar_index("image_hash", index_type="BTREE")
                self.log_diag(f"[DB] Created scalar indices on 'content_key' and 'image_hash'.")
                
                # Since we created a new one, update the combo box
                self.table_combo.addItem(target_table)
                self.table_combo.setCurrentText(target_table)
            
            return True
        except Exception as e:
            self.log_diag(f"[DB ERROR] {e}")
            return False

    def create_new_table(self):
        new_name = self.table_input.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Error", "Please provide a valid table name.")
            return
        self.log_diag(f"[DB] Attempting to create new table: {new_name}")
        self.connect_db(specific_table=new_name, force_create=True)

    def scan_for_tables(self):
        work_dir = self.dir_input.text()
        try:
            temp_db = lancedb.connect(work_dir)
            tables = temp_db.list_tables().tables
            self.table_combo.blockSignals(True)
            self.table_combo.clear()
            self.table_combo.addItems(tables)
            self.table_combo.blockSignals(False)
            self.log_diag(f"[DB] Scanned directory. Found {len(tables)} tables.")
        except Exception as e:
            self.log_diag(f"[DB ERROR] Failed to scan tables: {e}")

    def load_selected_table(self):
        selected = self.table_combo.currentText()
        if selected:
            self.log_diag(f"[DB] User triggered load for table: {selected}")
            self.connect_db(specific_table=selected, force_create=False)
        else:
            self.log_diag("[DB ERROR] No table selected in dropdown.")

    def build_fts_index(self):
        """Creates or updates the Full-Text Search index on the text column."""
        if self.table is None:
            if not self.connect_db(): return
            
        try:
            self.log_diag("[DB] Rebuilding FTS Index on 'text' column. Please wait...")
            self.build_fts_btn.setEnabled(False)
            QApplication.processEvents()
            
            self.table.create_fts_index("text", replace=True)
            
            self.log_diag("[DB] FTS Index successfully built/updated!")
        except Exception as e:
            self.log_diag(f"[DB ERROR] Failed to build FTS Index: {e}")
        finally:
            self.build_fts_btn.setEnabled(True)

    def build_vector_index(self):
        """Constructs the ANN vector index after bulk ingestion is complete."""
        if self.table is None:
            if not self.connect_db(): return
            
        try:
            self.log_diag("[DB] Building Vector Index on 'video_vector'. Please wait...")
            self.build_vec_btn.setEnabled(False)
            QApplication.processEvents()
            
            self.table.create_index(
                vector_column_name="video_vector",
                metric="cosine", 
                index_type="IVF_HNSW_SQ",
                num_partitions=256, # Optimized for ~2M rows to keep centroids in CPU cache
                replace=True
            )
            
            self.log_diag("[DB] Vector Index successfully built/updated!")
        except Exception as e:
            self.log_diag(f"[DB ERROR] Failed to build Vector Index: {e}")
        finally:
            self.build_vec_btn.setEnabled(True)



    @staticmethod
    def _calc_hash(data: str) -> str:
        """Calculates SHA256 hash for text/metadata."""
        if not data: return ""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    @staticmethod
    def _calc_image_hash(image_path: str) -> str:
        """Calculates SHA256 hash for image file bytes."""
        if not image_path or not os.path.exists(image_path): return ""
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _check_and_clean_duplicates(self, text_hash: str, img_hash: str, meta_hash: str):
        """Stage 2: Find and delete full duplicates (same text, image, metadata hashes)."""
        conditions = []
        if text_hash: conditions.append(f"text_hash = '{text_hash}'")
        if img_hash: conditions.append(f"image_hash = '{img_hash}'")
        if meta_hash: conditions.append(f"metadata_hash = '{meta_hash}'")
        
        if not conditions: return False
        
        where_clause = " AND ".join(conditions)
        try:
            dupes = self.table.search().where(where_clause).limit(10).to_pandas()
            if len(dupes) >= 1:
                ids_to_delete = dupes['id'].tolist()[1:]
                if ids_to_delete:
                    ids_str = ", ".join([f"'{uid}'" for uid in ids_to_delete])
                    self.table.delete(f"id IN ({ids_str})")
                    self.log_diag(f"[DEDUP] Cleaned {len(ids_to_delete)} full duplicate(s) for hash combo.")
                return True
            return False
        except Exception as e:
            self.log_diag(f"[DEDUP ERROR] {e}")
            return False

    def _check_and_clean_allduplicates(self, text_hash: str, img_hash: str, meta_hash: str):
        """Stage 2: Find and delete full duplicates (same text, image, metadata hashes)."""
        conditions = []
        if text_hash: conditions.append(f"text_hash = '{text_hash}'")
        if img_hash: conditions.append(f"image_hash = '{img_hash}'")
        if meta_hash: conditions.append(f"metadata_hash = '{meta_hash}'")
        
        if not conditions: return False
        
        where_clause = " AND ".join(conditions)
        try:
            dupes = self.table.search().where(where_clause).limit(10).to_pandas()
            if len(dupes) >= 1:
                ids_to_delete = dupes['id'].tolist()
                if ids_to_delete:
                    ids_str = ", ".join([f"'{uid}'" for uid in ids_to_delete])
                    self.table.delete(f"id IN ({ids_str})")
                    self.log_diag(f"[DEDUP] Cleaned {len(ids_to_delete)} full duplicate(s) for hash combo.")
                return True
            return False
        except Exception as e:
            self.log_diag(f"[DEDUP ERROR] {e}")
            return False

    def _batch_clean_duplicates(self, unique_pairs: list[tuple[str, str]]):
        """Stage 2 Optimized: Find and delete full duplicates safely via collection filtering."""
        if not unique_pairs:
            return

        # Extract discrete lists of unique hashes present in this entire batch
        text_hashes_to_clear = list(set([th for th, _ in unique_pairs if th]))
        meta_hashes_to_clear = list(set([mh for _, mh in unique_pairs if mh]))

        if not text_hashes_to_clear:
            return

        try:
            # Build SQL-compliant comma-separated sets
            th_placeholders = ", ".join([f"'{h}'" for h in text_hashes_to_clear])
            
            if meta_hashes_to_clear:
                mh_placeholders = ", ".join([f"'{h}'" for h in meta_hashes_to_clear])
                # Intersect conditions safely to guarantee precise column alignment match across sets
                where_clause = f"text_hash IN ({th_placeholders}) AND metadata_hash IN ({mh_placeholders})"
            else:
                where_clause = f"text_hash IN ({th_placeholders})"

            self.table.delete(where_clause)
            self.log_diag(f"[DEDUP] Swept existing database items matching batch signature lists.")
        except Exception as e:
            self.log_diag(f"[DEDUP ERROR] Batch clean failed: {e}")

    def _check_and_clean_dup_ti(self, text_hash: str, img_hash: str):
        """Stage 2b: Find and delete text+image duplicates (same text, image)."""
        conditions = []
        if text_hash: conditions.append(f"text_hash = '{text_hash}'")
        if img_hash: conditions.append(f"image_hash = '{img_hash}'")
        
        
        if not conditions: return False
        
        where_clause = " AND ".join(conditions)
        try:
            dupes = self.table.search().where(where_clause).limit(10).to_pandas()
            if len(dupes) >= 1:
                ids_to_delete = dupes['id'].tolist()[1:]
                if ids_to_delete:
                    ids_str = ", ".join([f"'{uid}'" for uid in ids_to_delete])
                    self.table.delete(f"id IN ({ids_str})")
                    self.log_diag(f"[DEDUP] Cleaned {len(ids_to_delete)} full duplicate(s) for hash combo.")
                return True
            return False
        except Exception as e:
            self.log_diag(f"[DEDUP ERROR] {e}")
            return False


    def _core_ingest(self, text: str, img_path: str, meta: str):
        """ The unified insertion method used by both Single and Batch flows """
        if self.table is None:
            if not self.connect_db(): return

        try:
            active_cfg = self.model_combo.currentData()
            m_type = active_cfg.get("type")
            dim = active_cfg.get("dimension")
            
            text_hash = self._calc_hash(text) if text and text.strip() else ""
            img_hash = self._calc_image_hash(img_path) if img_path and os.path.exists(img_path) else ""
            meta_hash = self._calc_hash(meta) if meta else ""
            
            # Stage 2: Cleanup full duplicates before insertion
            self._check_and_clean_allduplicates(text_hash, img_hash, meta_hash)
            
            text_vec = [0.0] * dim
            img_vec = [0.0] * dim
            
            # Stage 1: Lookup existing hashes to skip expensive embedding
            img_reused = False
            txt_reused = False
            
            if img_hash:
                match = self.table.search().where(f"image_hash = '{img_hash}'").limit(1).to_pandas()
                if len(match) > 0:
                    img_vec = match['image_vector'].iloc[0]
                    img_reused = True
                    self.log_diag(f"[DEDUP] Image hash hit. Reused vector.")

            if text_hash:
                match = self.table.search().where(f"text_hash = '{text_hash}'").limit(1).to_pandas()
                if len(match) > 0:
                    text_vec = match['text_vector'].iloc[0]
                    txt_reused = True
                    self.log_diag(f"[DEDUP] Text hash hit. Reused vector.")
            
            # Only run model if vectors are missing
            if not img_reused or not txt_reused:
                if m_type == "dummy" or self.active_model_obj == "dummy":
                    if not txt_reused and text and text.strip():
                        v = np.random.rand(dim).astype(np.float32)
                        text_vec = (v / np.linalg.norm(v)).tolist()
                    if not img_reused and img_path and os.path.exists(img_path):
                        v = np.random.rand(dim).astype(np.float32)
                        img_vec = (v / np.linalg.norm(v)).tolist()
                else:
                    import torch
                    from PIL import Image
                    with torch.no_grad():
                        self.active_model_obj.eval()
                        inputs = []
                        if not txt_reused and text and text.strip(): inputs.append(text)
                        if not img_reused and img_path and os.path.exists(img_path): inputs.append(Image.open(img_path).convert("RGB"))
                        
                        if inputs:
                            features = self.active_model_obj.encode(inputs)
                            if hasattr(features, "detach"):
                                features = features.detach().cpu().numpy()
                            features = np.array(features, dtype=np.float32)
                            norms = np.linalg.norm(features, axis=1, keepdims=True)
                            norms[norms == 0] = 1 
                            features = features / norms
                            
                            idx = 0
                            if not txt_reused and text and text.strip():
                                text_vec = features[idx].tolist()
                                idx += 1
                            if not img_reused and img_path and os.path.exists(img_path):
                                img_vec = features[idx].tolist()
                                idx += 1

            # Validation
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in text_vec):
                self.log_diag(f"[DB] Text vector for {os.path.basename(img_path) if img_path else 'snippet'} contains NaN. Using zeros.")
                text_vec = [0.0] * dim
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in img_vec):
                self.log_diag(f"[DB] Image vector for {os.path.basename(img_path) if img_path else 'snippet'} contains NaN. Using zeros.")
                img_vec = [0.0] * dim

            # NEW: Compute empty flags for search-time filtering
            is_empty_text = all(v == 0.0 for v in text_vec)
            is_empty_image = all(v == 0.0 for v in img_vec)

            row = [{
                "id": str(uuid.uuid4()),
                "text_vector": text_vec,
                "image_vector": img_vec,
                "video_vector": [0.0] * dim,
                "text": text,
                "image_path": img_path,
                "metadata": meta,
                "text_hash": text_hash,
                "image_hash": img_hash,
                "metadata_hash": meta_hash,
                "empty_text": is_empty_text,
                "empty_image": is_empty_image,
                "empty_video": True,
                "type": "standard",
                "fps": 0.0,
                "content_key": text_hash
            }]



            self.table.add(row)
            self.log_diag(f"[DB] Inserted: {os.path.basename(img_path) if img_path else 'Text snippet'}")
            
        except Exception as e:
            self.log_diag(f"[DB ERROR] Failed to ingest {img_path}: {e}")

    def _flush_buffer(self, ingest_key: str, row_buffer: list[dict]):
        """Flushes buffer using fast O(1) appends. Deduplication handled in-memory."""
        if not row_buffer: return
        try:
            # Deduplicate within buffer to prevent intra-batch duplicates
            seen = set()
            unique_rows = []
            for row in row_buffer:
                k = row.get(ingest_key)
                if k not in seen:
                    seen.add(k)
                    unique_rows.append(row)
            
            if not unique_rows:
                row_buffer.clear()
                return

            arrow_table = pa.Table.from_pylist(unique_rows)
            self.table.add(arrow_table) # Fast append instead of merge_insert
            
            # REMOVED: compact_files() and cleanup_old_versions() from hot path.
            # Calling these every flush causes O(N^2) SSD writes and RSS inflation.
            # Maintenance is now centralized in _finalize_ingestion().
            
            flushed_count = len(unique_rows)
            row_buffer.clear()
            gc.collect()
            self.log_diag(f"[DB] Appended {flushed_count} rows to table. ")
        except Exception as e:
            self.log_diag(f"[DB ERROR] Buffer flush failed: {e} ")

    def _accumulate_and_flush(self, key_column: str, new_rows: list):
        """Appends rows to the global buffer and flushes when the record limit is reached."""
        #import gc
        if not new_rows:
            return
        if self.ingest_buffer_key and self.ingest_buffer_key != key_column:
            self._flush_buffer(self.ingest_buffer_key, self.ingest_buffer)
            self.ingest_buffer = []
        self.ingest_buffer_key = key_column
        self.ingest_buffer.extend(new_rows)
        limit = self.batch_buffer_size.value()
        ### diagnostics on terminal
        print(f"ingest_buffer status: {len(self.ingest_buffer)} / {limit}")
        if len(self.ingest_buffer) >= limit:
            self._flush_buffer(key_column, self.ingest_buffer)
            # Force GC after flushing the global buffer to prevent memory buildup
            gc.collect()


    def _flush_final_buffer(self):
        """Flushes any remaining records in the buffer at the end of an ingestion cycle."""
        if self.ingest_buffer:
            self._flush_buffer(self.ingest_buffer_key, self.ingest_buffer)

    def _finalize_ingestion(self):
        """Flushes any remaining buffers and performs heavy disk maintenance ONCE."""
        if self.ingest_buffer:
            self._flush_buffer(self.ingest_buffer_key, self.ingest_buffer)
            
        self.log_diag("[DB] Ingestion batch finished. Optimizing table structure...")
        try:
            self.table.compact_files()
            import datetime
            self.table.cleanup_old_versions(older_than=datetime.timedelta(minutes=0), delete_unverified=True)
            self.log_diag("[DB] Compaction and version cleanup complete.")
        except Exception as e:
            self.log_diag(f"[DB ERROR] Maintenance failed: {e}")

    def _embed_and_buffer_texts(self, texts: list[str], metas: list[str]):
        """Decoupled text embedding & buffering workflow."""
        if self.table is None:
            if not self.connect_db(): return

        try:
            active_cfg = self.model_combo.currentData()
            dim = active_cfg.get("dimension")
            m_type = active_cfg.get("type")

            # Deduplicate within chunk
            seen_hashes = set()
            unique_indices = []
            for i, t in enumerate(texts):
                th = self._calc_hash(t)
                if th not in seen_hashes:
                    seen_hashes.add(th)
                    unique_indices.append(i)

            if not unique_indices:
                return

            unique_texts = [texts[i] for i in unique_indices]
            unique_metas = [metas[i] for i in unique_indices]
            unique_hashes = [self._calc_hash(t) for t in unique_texts]

            # Check DB for existing vectors
            hash_to_vec = {}
            placeholders = ", ".join([f"'{h}'" for h in unique_hashes])
            matches = self.table.search().where(f"text_hash IN ({placeholders})").to_pandas()
            for _, row in matches.iterrows():
                hash_to_vec[row['text_hash']] = row['text_vector']

            # Identify texts needing embedding
            texts_to_embed = [t for t, h in zip(unique_texts, unique_hashes) if h not in hash_to_vec]
            computed_vectors = {}

            if texts_to_embed:
                if m_type == "dummy" or self.active_model_obj == "dummy":
                    for t in texts_to_embed:
                        v = np.random.rand(dim).astype(np.float32)
                        computed_vectors[t] = (v / np.linalg.norm(v)).tolist()
                else:
                    import torch
                    with torch.no_grad():
                        self.active_model_obj.eval()
                        features = self.active_model_obj.encode(texts_to_embed)
                        if hasattr(features, "detach"):
                            features = features.detach().cpu().numpy()
                        features = np.array(features, dtype=np.float32)
                        norms = np.linalg.norm(features, axis=1, keepdims=True)
                        norms[norms == 0] = 1
                        features = features / norms
                        for idx, t in enumerate(texts_to_embed):
                            computed_vectors[t] = features[idx].tolist()

            # Construct rows and buffer them
            rows = []
            for t, m, h in zip(unique_texts, unique_metas, unique_hashes):
                if h in hash_to_vec:
                    continue # Skip texts that already exist in the database
                vec = hash_to_vec.get(h, computed_vectors.get(t, [0.0] * dim))
                rows.append({
                    "id": str(uuid.uuid4()),
                    "text_vector": vec,
                    "image_vector": [0.0] * dim,
                    "video_vector": [0.0] * dim,
                    "text": t,
                    "image_path": "",
                    "metadata": m,
                    "text_hash": h,
                    "image_hash": "",
                    "metadata_hash": self._calc_hash(m),
                    "empty_text": False,
                    "empty_image": True,
                    "empty_video": True,
                    "type": "standard",
                    "fps": 0.0,
                    "content_key": h
                })
            self._accumulate_and_flush("text_hash", rows)

        except Exception as e:
            self.log_diag(f"[DB ERROR] Text embedding/buffer failed: {e}")


    def _core_batch_image_ingest(self, img_paths: list[str]):
        if self.table is None:
            if not self.connect_db(): return

        try:
            active_cfg = self.model_combo.currentData()
            dim = active_cfg.get("dimension")
            m_type = active_cfg.get("type")
            embed_batch_size = self.image_batch_size.value()
            self.log_diag(f"[IMAGE] Starting batch ingestion of {len(img_paths)} images (embed batch size={embed_batch_size}).")

            img_hashes = [self._calc_image_hash(p) for p in img_paths]
            text_hashes = [""] * len(img_paths)
            meta_list = [json.dumps({"source": os.path.basename(p), "type": "image_batch"}) for p in img_paths]
            meta_hashes = [self._calc_hash(m) for m in meta_list]
            
            # Cache lookup
            unique_hashes = list(set([h for h in img_hashes if h]))
            hash_to_vec = {}
            if unique_hashes:
                placeholders = ", ".join([f"'{h}'" for h in unique_hashes])
                matches = self.table.search().where(f"image_hash IN ({placeholders})").to_pandas()
                for _, row in matches.iterrows():
                    hash_to_vec[row['image_hash']] = row['image_vector']
            
            # Identify paths needing embedding
            paths_to_embed = []
            for i, h in enumerate(img_hashes):
                if h and h not in hash_to_vec:
                    paths_to_embed.append(img_paths[i])
            
            computed_vectors = {}
            if paths_to_embed:
                if m_type == "dummy" or self.active_model_obj == "dummy":
                    for p in paths_to_embed:
                        v = np.random.rand(dim).astype(np.float32)
                        computed_vectors[p] = (v / np.linalg.norm(v)).tolist()
                    self.log_diag(f"[IMAGE] Generated {len(paths_to_embed)} dummy vectors.")
                else:
                    import torch
                    from PIL import Image
                    # Process in configurable embedding batches
                    total_embed_batches = (len(paths_to_embed) + embed_batch_size - 1) // embed_batch_size
                    for batch_idx, i in enumerate(range(0, len(paths_to_embed), embed_batch_size), start=1):
                        chunk_paths = paths_to_embed[i:i+embed_batch_size]
                        with torch.no_grad():
                            self.active_model_obj.eval()
                            images = [Image.open(p).convert("RGB") for p in chunk_paths]
                            features = self.active_model_obj.encode(images)
                            if hasattr(features, "detach"):
                                features = features.detach().cpu().numpy()
                            features = np.array(features, dtype=np.float32)
                            norms = np.linalg.norm(features, axis=1, keepdims=True)
                            norms[norms == 0] = 1
                            features = features / norms
                            for idx, p in enumerate(chunk_paths):
                                computed_vectors[p] = features[idx].tolist()
                        remaining_imgs = len(paths_to_embed) - (batch_idx * embed_batch_size)
                        self.log_diag(f"[IMAGE] Embedded batch {batch_idx}/{total_embed_batches} ({len(chunk_paths)} images, {max(0, remaining_imgs)} remaining) ")

            # Build rows and buffer them
            for path, ih, th, mh in zip(img_paths, img_hashes, text_hashes, meta_hashes):
                if ih in hash_to_vec:
                    continue # Skip images that already exist in the database
                elif path in computed_vectors:
                    i_vec = computed_vectors[path]
                else:
                    i_vec = [0.0] * dim
                t_vec = [0.0] * dim
                meta = json.dumps({"source": os.path.basename(path), "type": "image_batch"})
                self._accumulate_and_flush("image_hash", [{
                    "id": str(uuid.uuid4()),
                    "text_vector": t_vec,
                    "image_vector": i_vec,
                    "text": "",
                    "image_path": path,
                    "metadata": meta,
                    "text_hash": th,
                    "image_hash": ih,
                    "metadata_hash": mh,
                    "empty_text": True,
                    "empty_image": all(v == 0.0 for v in i_vec),
                    "empty_video": True,
                    "type": "standard",
                    "fps": 0.0,
                    "content_key": ih
                }])
            
            self.log_diag(f"[DB] Inserted vectorized image batch via buffered merge_insert.")
            
        except Exception as e:
            self.log_diag(f"[DB ERROR] Failed to ingest image batch: {e}")



    def ingest_data(self):
        text = self.ingest_text.text()
        img_path = self.ingest_img.text()
        meta = self.ingest_meta.text()

        if not text and not img_path:
            QMessageBox.warning(self, "Input Error", "Provide text or image path.")
            return

        self.log_diag("[DB] Generating embedding for single item...")
        QApplication.processEvents() 
        self._core_ingest(text, img_path, meta)
        
        self.ingest_text.clear()
        self.ingest_img.clear()

    # =====================================================================
    # BATCH & LIVE MONITORING LOGIC
    # =====================================================================

    def stop_ongoing_ingestion(self):
        self.stop_requested = True
        self.log_diag("[SYSTEM] Stop requested. Finishing current atomic operations...")
        self.stop_ingest_btn.setEnabled(False)

    def _probe_video(self, video_path: str):
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "v:0", video_path
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            data = json.loads(result.stdout)
            if 'streams' in data and len(data['streams']) > 0:
                stream = data['streams'][0]
                fps_str = stream.get('r_frame_rate', '0/1')
                num, den = map(int, fps_str.split('/'))
                native_fps = num / den if den != 0 else 30.0
                
                duration = float(stream.get('duration', 0))
                if duration == 0:
                    cmd_fmt = [
                        "ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", video_path
                    ]
                    res_fmt = subprocess.run(cmd_fmt, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                    fmt_data = json.loads(res_fmt.stdout)
                    duration = float(fmt_data.get('format', {}).get('duration', 0))
                
                return native_fps, duration
        except Exception as e:
            self.log_diag(f"[PROBE ERROR] {e}")
        return 30.0, 0.0


    def set_gui_enabled(self, enabled: bool):
        """
        Locks or unlocks interactive elements to prevent database corruption from 
        out-of-band requests while synchronous ingestion yields to the main event loop.
        """
        self.is_ingesting = not enabled
        self.ingest_tabs.setEnabled(enabled)
        self.load_model_btn.setEnabled(enabled)
        self.start_srv_btn.setEnabled(enabled)
        self.query_btn.setEnabled(enabled)
        self.stop_ingest_btn.setEnabled(self.is_ingesting)
        
        # Lock new table controls
        self.create_table_btn.setEnabled(enabled)
        self.scan_tables_btn.setEnabled(enabled)
        self.load_table_btn.setEnabled(enabled)
        self.table_combo.setEnabled(enabled)
        
        if not enabled:
            self.log_diag("\n" + "="*50)
            self.log_diag("[SYSTEM] PLEASE WAIT - GUI LOCKED TO AVOID DATA CORRUPTION DURING INGESTION")
            self.log_diag("="*50 + "\n")
        else:
            self.log_diag("\n[SYSTEM] INGESTION COMPLETE - GUI UNLOCKED\n")

    def start_batch_ingest(self):
        directory = self.batch_dir_input.text()
        if not directory or not os.path.exists(directory):
            QMessageBox.warning(self, "Error", "Invalid directory selected.")
            return

        self.log_diag(f"[BATCH] Starting batch ingest in {directory}...")
        self.set_gui_enabled(False)
        
        exts = ['.png', '.jpg', '.jpeg', '.tiff', '.webp']
        if self.chk_pdf.isChecked():
            exts.append('.pdf')
        if hasattr(self, 'chk_csv') and self.chk_csv.isChecked():
            exts.append('.csv')
        if hasattr(self, 'chk_archive') and self.chk_archive.isChecked():
            exts.extend(['.zip', '.tar.gz', '.tgz', '.7z'])
        
        if not self.chk_assoc.isChecked():
            exts.extend(['.txt', '.json', '.py', '.c', '.h', '.html', '.md', '.js', '.cpp'])
            
        found_files = []
        for root, dirs, files in os.walk(directory):
            if self.stop_requested: break
            for file in files:
                lower_file = file.lower()
                matched = False
                for ext in exts:
                    if lower_file.endswith(ext):
                        found_files.append(os.path.join(root, file))
                        matched = True
                        break
            if not self.chk_recurse.isChecked():
                break
    
        if self.stop_requested:
            self.log_diag("[SYSTEM] Ingestion stopped by user.")
            self.set_gui_enabled(True)
            return

        if found_files:
            self.process_file_batch(found_files)
        else:
            self.log_diag("[BATCH] No valid target files found.")
            
        self.set_gui_enabled(True)

    def has_associated_metadata(self, path: str) -> bool:
        """ Checks if the image has matching .txt or .json files in its directory """
        if not self.chk_assoc.isChecked():
            return False
            
        img_dir = os.path.dirname(path)
        img_base = os.path.splitext(os.path.basename(path))[0]
        try:
            for f in os.listdir(img_dir):
                f_ext = os.path.splitext(f)[1].lower()
                f_base = os.path.splitext(f)[0]
                if f_ext in {'.txt', '.json'} and img_base in f_base:
                    return True
        except Exception:
            pass
        return False

    def convert_webp_to_png(self, path: str) -> str:
        """ Converts a webp image to png with a unique suffix so it won't be re-converted. """
        try:
            from PIL import Image
            import hashlib
            
            uid = hashlib.md5(path.encode('utf-8')).hexdigest()[:8]
            img_dir = os.path.dirname(path)
            img_base = os.path.splitext(os.path.basename(path))[0]
            new_name = f"{img_base}_conv_{uid}.png"
            new_path = os.path.join(img_dir, new_name)
            
            if not os.path.exists(new_path):
                with Image.open(path) as img:
                    img.convert("RGB").save(new_path, "PNG")
                self.log_diag(f"[CONVERT] WebP to PNG: {new_name}")
            return new_path
        except Exception as e:
            self.log_diag(f"[CONVERT ERROR] Failed to convert {path}: {e}")
            return ""

    def batch_process_texts(self, file_paths: list):
        batch_texts = []
        batch_metas = []
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                if text:
                    batch_texts.append(text)
                    batch_metas.append(json.dumps({"source": os.path.basename(path), "type": "code_or_text"}))
            except Exception as e:
                self.log_diag(f"[TEXT ERROR] Failed to read {path}: {e}")

        if not batch_texts:
            return
        self.log_diag(f"[TEXT] Starting batch ingestion of {len(batch_texts)} texts from {len(file_paths)} files.")

        chunk_size = self.text_batch_size.value()
        total_chunks = (len(batch_texts) + chunk_size - 1) // chunk_size
        for chunk_idx, i in enumerate(range(0, len(batch_texts), chunk_size), start=1):
            if self.stop_requested: break
            chunk_texts = batch_texts[i:i+chunk_size]
            chunk_metas = batch_metas[i:i+chunk_size]
            self._embed_and_buffer_texts(chunk_texts, chunk_metas)
            remaining = len(batch_texts) - (i + len(chunk_texts))
            self.log_diag(f"[TEXT] Embedded chunk {chunk_idx}/{total_chunks} ({len(chunk_texts)} texts, {max(0, remaining)} remaining)")
            QApplication.processEvents()

        self.log_diag(f"[TEXT] Buffered ingestion complete for {len(batch_texts)} texts.")


    def batch_process_images(self, img_paths: list):
        if not img_paths: return
        self._core_batch_image_ingest(img_paths)
        QApplication.processEvents()


    @Slot(list)
    def process_file_batch(self, file_paths: list):
        text_exts = {'.txt', '.json', '.py', '.c', '.h', '.html', '.md', '.js', '.cpp'}
        
        # Prevent unbounded memory growth from tracking processed files over long sessions
        if len(self.processed_files) > 10000:
            self.processed_files.clear()

        text_files_to_batch = []
        image_files_to_batch = []
        image_files_single = []
        other_files = []

        for path in file_paths:
            if self.stop_requested: break
            if path in self.processed_files:
                continue
                
            lower_path = path.lower()
            
            if lower_path.endswith('.webp'):
                orig_path = path
                path = self.convert_webp_to_png(path)
                self.processed_files.add(orig_path)
                if not path:
                    continue
                lower_path = path.lower()

            if lower_path.endswith('.pdf'):
                other_files.append(('pdf', path))
            elif lower_path.endswith('.csv') and hasattr(self, 'chk_csv') and self.chk_csv.isChecked():
                other_files.append(('csv', path))
            elif lower_path.endswith(('.zip', '.tar.gz', '.tgz', '.7z')) and hasattr(self, 'chk_archive') and self.chk_archive.isChecked():
                other_files.append(('archive', path))
            elif lower_path.endswith(tuple(text_exts)):
                text_files_to_batch.append(path)
            elif lower_path.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                if self.has_associated_metadata(path):
                    image_files_single.append(path)
                else:
                    image_files_to_batch.append(path)

        # 1. Process Others (PDF, CSV, Archive)
        for ftype, path in other_files:
            if ftype == 'pdf': self.process_pdf_file(path)
            elif ftype == 'csv': self.process_csv_file(path)
            elif ftype == 'archive': self.process_archive_file(path)
            self.processed_files.add(path)
            QApplication.processEvents()

        # 2. Process Images with Metadata (Single Pipeline)
        for path in image_files_single:
            self.process_image_file(path)
            self.processed_files.add(path)
            QApplication.processEvents()

        # 3. Process Text Files (Decoupled Buffering Workflow)
        if text_files_to_batch:
            self.batch_process_texts(text_files_to_batch)
            for path in text_files_to_batch:
                self.processed_files.add(path)

        # 4. Process Images without Metadata (Decoupled Buffering Workflow)
        if image_files_to_batch:
            self.batch_process_images(image_files_to_batch)
            for path in image_files_to_batch:
                self.processed_files.add(path)

        self._finalize_ingestion()        
        self.log_diag("[BATCH/LIVE] Processing cycle complete.")
    
    # --- NEW CODE: CSV Processor ---
    def process_csv_file(self, path: str):
        import csv
        try:
            window_size = self.csv_window_size.value() if hasattr(self, 'csv_window_size') else 256
            if (window_size > 500): # LanceDB has a hard limit of 500 WHERE conditions as of May, 2026. Exceeding this value will throw an error. Exceeding it by far, will cause a segfault.
                window_size = 500
            batch_texts = []
            batch_metas = []
            
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                headers = None
                
                if hasattr(self, 'chk_csv_meta') and self.chk_csv_meta.isChecked():
                    headers = next(reader, None)
                    
                for i, row in enumerate(reader):
                    if self.stop_requested: break
                    if not row: continue
                    text = " | ".join(row).strip()
                    if not text: continue
                    
                    meta_dict = {
                        "source": os.path.basename(path), 
                        "row_index": i + (2 if headers else 1), 
                        "type": "csv_record"
                    }
                    
                    if headers:
                        for idx, val in enumerate(row):
                            if idx < len(headers):
                                cleaned_val = val.strip()
                                if cleaned_val.isdigit():
                                    meta_dict[headers[idx].strip()] = int(cleaned_val)
                                else:
                                    try:
                                        meta_dict[headers[idx].strip()] = float(cleaned_val)
                                    except ValueError:
                                        meta_dict[headers[idx].strip()] = cleaned_val

                    batch_texts.append(text)
                    batch_metas.append(json.dumps(meta_dict))
                    ### diagnostics on terminal
                    #print(f"CSV window status: {len(batch_texts)} / {window_size}")
                    if len(batch_texts) >= window_size:
                        self._embed_and_buffer_texts(batch_texts, batch_metas)
                        batch_texts.clear()
                        batch_metas.clear()
                        # Yield control after every window to maintain responsiveness
                        QApplication.processEvents()

                # Flush any remaining items in the buffer
                if batch_texts:
                    self._embed_and_buffer_texts(batch_texts, batch_metas)
                    QApplication.processEvents()
                    
        except Exception as e:
            self.log_diag(f"[CSV ERROR] Failed to ingest {path}: {e}")

    # -------------------------------



    def process_text_file(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            if text:
                meta = json.dumps({"source": os.path.basename(path), "type": "code_or_text"})
                self._core_ingest(text, "", meta)
        except Exception as e:
            self.log_diag(f"[TEXT ERROR] Failed to ingest {path}: {e}")

    def process_image_file(self, path: str):
        if self.table is None:
            if not self.connect_db(): return

        active_cfg = self.model_combo.currentData()
        dim = active_cfg.get("dimension")
        m_type = active_cfg.get("type")
        
        text = ""
        meta_dict = {"source": os.path.basename(path), "type": "image_single"}
        
        if self.chk_assoc.isChecked():
            img_dir = os.path.dirname(path)
            img_name = os.path.basename(path)
            img_base = os.path.splitext(img_name)[0]

            try:
                all_files = os.listdir(img_dir)
                
                for f in all_files:
                    f_ext = os.path.splitext(f)[1].lower()
                    f_base = os.path.splitext(f)[0]
                    
                    if f_ext in {'.txt', '.json'} and img_base in f_base:
                        companion_path = os.path.join(img_dir, f)
                        try:
                            with open(companion_path, 'r', encoding='utf-8', errors='ignore') as c_file:
                                content = c_file.read().strip()
                                if content:
                                    text = content
                                    self.log_diag(f"[ASSOCIATE] Matched companion text/metadata: {f} to {img_name}")
                                    break 
                        except Exception as e:
                            self.log_diag(f"[ASSOCIATE ERROR] Failed to read {f}: {e}")
            except Exception as e:
                self.log_diag(f"[ASSOCIATE ERROR] Directory scan failed for {img_dir}: {e}")

        meta_str = json.dumps(meta_dict)
        
        try:
            # Compute hashes
            img_hash = self._calc_image_hash(path)
            text_hash = self._calc_hash(text) if text else ""
            
            # Look up existing vectors in DB to avoid re-embedding
            hash_to_vec = {}
            if img_hash:
                match = self.table.search().where(f"image_hash = '{img_hash}'").limit(1).to_pandas()
                if len(match) > 0:
                    self.log_diag(f"[DEDUP] Image {os.path.basename(path)} already exists in DB. Skipping.")
                    return
                    
            if text_hash:
                match = self.table.search().where(f"text_hash = '{text_hash}'").limit(1).to_pandas()
                if len(match) > 0:
                    hash_to_vec['text'] = match['text_vector'].iloc[0]

            # Embed what's missing
            img_vec = [0.0] * dim
            text_vec = [0.0] * dim
            
            if 'image' not in hash_to_vec and img_hash:
                if m_type == "dummy" or self.active_model_obj == "dummy":
                    v = np.random.rand(dim).astype(np.float32)
                    img_vec = (v / np.linalg.norm(v)).tolist()
                else:
                    import torch
                    from PIL import Image
                    with torch.no_grad():
                        self.active_model_obj.eval()
                        features = self.active_model_obj.encode(Image.open(path).convert("RGB"))
                        if hasattr(features, "detach"):
                            features = features.detach().cpu().numpy()
                        features = np.array(features, dtype=np.float32)
                        norm = np.linalg.norm(features)
                        if norm > 0:
                            img_vec = (features / norm).tolist()

            if 'text' not in hash_to_vec and text_hash:
                if m_type == "dummy" or self.active_model_obj == "dummy":
                    v = np.random.rand(dim).astype(np.float32)
                    text_vec = (v / np.linalg.norm(v)).tolist()
                else:
                    import torch
                    with torch.no_grad():
                        self.active_model_obj.eval()
                        features = self.active_model_obj.encode(text)
                        if hasattr(features, "detach"):
                            features = features.detach().cpu().numpy()
                        features = np.array(features, dtype=np.float32)
                        norm = np.linalg.norm(features)
                        if norm > 0:
                            text_vec = (features / norm).tolist()

            # Reuse from cache if found
            if 'image' in hash_to_vec:
                img_vec = hash_to_vec['image']
            if 'text' in hash_to_vec:
                text_vec = hash_to_vec['text']

            # Construct record with image_hash as the primary key for upsert
            record = {
                "id": str(uuid.uuid4()),
                "text_vector": text_vec,
                "image_vector": img_vec,
                "video_vector": [0.0] * dim,
                "text": text,
                "image_path": path,
                "metadata": meta_str,
                "text_hash": text_hash,
                "image_hash": img_hash,
                "metadata_hash": self._calc_hash(meta_str),
                "empty_text": not bool(text),
                "empty_image": all(v == 0.0 for v in img_vec),
                "empty_video": True,
                "type": "standard",
                "fps": 0.0,
                "content_key": img_hash 
            }
            
            self._accumulate_and_flush("image_hash", [record])
            
        except Exception as e:
            self.log_diag(f"[DB ERROR] Failed to ingest {path}: {e}")

    def process_pdf_file(self, path: str):
        try:
            import fitz # PyMuPDF
        except ImportError:
            self.log_diag("[ERROR] PyMuPDF not installed. Cannot process PDF.")
            self.log_diag("--> run: pip install pymupdf")
            return

        self.log_diag(f"[PDF] Processing {os.path.basename(path)}...")
        
        if self.table is None:
            if not self.connect_db(): return

        try:
            active_cfg = self.model_combo.currentData()
            dim = active_cfg.get("dimension")
            m_type = active_cfg.get("type")

            # Prepare temp directory (prefer RAM disk for speed to prevent physical FS saturation)
            if os.path.exists("/dev/shm"):
                temp_dir = os.path.join("/dev/shm", f"pdf_extract_{uuid.uuid4().hex}")
            else:
                temp_dir = tempfile.mkdtemp(prefix="pdf_extract_")
            os.makedirs(temp_dir, exist_ok=True)

            doc = fitz.open(path)
            total_pages = len(doc)
            pdf_hash = self._calc_image_hash(path) # Reuses generic file byte hasher for content_key uniqueness
            
            # To prevent /dev/shm from filling up, process in batches of pages
            page_batch_size = self.pdf_batch_size.value()
            
            for page_start in range(0, total_pages, page_batch_size):
                if self.stop_requested:
                    break

                # Clean temp directory before extracting new batch to prevent RAM disk saturation
                for f in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, f))
                    except Exception:
                        pass

                # Extract batch
                page_buffer_data = [] # (page_num, text, img_path, content_key, meta_img)
                paths_to_embed_images = []
                texts_to_embed = []

                self.log_diag(f"[PDF] Processing from page {page_start} / {total_pages}")

                for i in range(page_start, min(page_start + page_batch_size, total_pages)):
                    page = doc[i]
                    text = page.get_text().strip()
                    
                    # Render page to image in RAM disk
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_path = os.path.join(temp_dir, f"page_{i+1:08d}.png")
                    pix.save(img_path)
                    pix = None # Free memory

                    content_key = f"{pdf_hash}_page_{i+1}"
                    meta_img = json.dumps({"source": os.path.basename(path), "page": i+1, "format": "visual_slice"})
                    
                    page_buffer_data.append((i+1, text, img_path, content_key, meta_img))
                    paths_to_embed_images.append(img_path)
                    if text:
                        texts_to_embed.append(text)

                # Check DB for existing vectors using Arrow for speed
                keys = [pd[3] for pd in page_buffer_data]
                safe_keys = [k.replace("'", "''") for k in keys]
                placeholders = ", ".join([f"'{k}'" for k in safe_keys])

                existing_matches = (
                    self.table.search()
                    .where(f"content_key IN ({placeholders})")
                    .select(["content_key", "image_vector", "text_vector"])
                    .to_arrow()
                )

                key_to_img_vec = {}
                key_to_txt_vec = {}
                if existing_matches is not None and len(existing_matches) > 0:
                    keys_col = existing_matches["content_key"].to_pylist()
                    img_vecs_col = existing_matches["image_vector"].to_pylist()
                    txt_vecs_col = existing_matches["text_vector"].to_pylist()
                    for k, iv, tv in zip(keys_col, img_vecs_col, txt_vecs_col):
                        key_to_img_vec[k] = iv
                        key_to_txt_vec[k] = tv
                del existing_matches

                # Determine what needs embedding
                imgs_to_embed = [pd[2] for pd in page_buffer_data if pd[3] not in key_to_img_vec]
                txts_to_embed = [pd[1] for pd in page_buffer_data if pd[3] not in key_to_txt_vec and pd[1]]

                computed_img_vecs = {}
                if imgs_to_embed:
                    if m_type == "dummy" or self.active_model_obj == "dummy":
                        for p in imgs_to_embed:
                            v = np.random.rand(dim).astype(np.float32)
                            computed_img_vecs[p] = (v / np.linalg.norm(v)).tolist()
                    else:
                        import torch
                        from PIL import Image
                        with torch.no_grad():
                            self.active_model_obj.eval()
                            images = []
                            valid_paths = []
                            for p in imgs_to_embed:
                                try:
                                    img = Image.open(p)
                                    img.load()
                                    images.append(img.convert("RGB"))
                                    valid_paths.append(p)
                                except Exception as img_err:
                                    self.log_diag(f"[PDF WARNING] Skipping corrupted/unreadable page image {os.path.basename(p)}: {img_err}")
                                    if 'img' in locals():
                                        try: img.close()
                                        except Exception: pass
                            
                            if images:
                                features = self.active_model_obj.encode(images)
                                if hasattr(features, "detach"):
                                    features = features.detach().cpu().numpy()
                                features = np.array(features, dtype=np.float32)
                                norms = np.linalg.norm(features, axis=1, keepdims=True)
                                norms[norms == 0] = 1
                                features = features / norms
                                for idx, p in enumerate(valid_paths):
                                    computed_img_vecs[p] = features[idx].tolist()
                            else:
                                self.log_diag("[PDF WARNING] No valid images to encode in this batch.")


                computed_txt_vecs = {}
                if txts_to_embed:
                    if m_type == "dummy" or self.active_model_obj == "dummy":
                        for t in txts_to_embed:
                            v = np.random.rand(dim).astype(np.float32)
                            computed_txt_vecs[t] = (v / np.linalg.norm(v)).tolist()
                    else:
                        import torch
                        with torch.no_grad():
                            self.active_model_obj.eval()
                            features = self.active_model_obj.encode(txts_to_embed)
                            if hasattr(features, "detach"):
                                features = features.detach().cpu().numpy()
                            features = np.array(features, dtype=np.float32)
                            norms = np.linalg.norm(features, axis=1, keepdims=True)
                            norms[norms == 0] = 1
                            features = features / norms
                            for idx, t in enumerate(txts_to_embed):
                                computed_txt_vecs[t] = features[idx].tolist()

                # Construct rows and clean up images immediately
                batch_rows = []
                for page_num, text, img_path, key, meta_img in page_buffer_data:

                        
                    # Skip pages that already exist in the database
                    if key in key_to_img_vec:
                        continue
                        
                    i_vec = key_to_img_vec.get(key, computed_img_vecs.get(img_path, [0.0] * dim))
                    t_vec = key_to_txt_vec.get(key, computed_txt_vecs.get(text, [0.0] * dim))
                    batch_rows.append({
                        "id": str(uuid.uuid4()),
                        "text_vector": t_vec,
                        "image_vector": i_vec,
                        "video_vector": [0.0] * dim,
                        "text": text,
                        "image_path": f"{path}_page_{page_num}", # Virtual path for on-demand recovery
                        "metadata": meta_img,
                        "text_hash": self._calc_hash(text) if text else "",
                        "image_hash": self._calc_image_hash(img_path),
                        "metadata_hash": self._calc_hash(meta_img),
                        "empty_text": not bool(text),
                        "empty_image": all(v == 0.0 for v in i_vec),
                        "empty_video": True,
                        "type": "pdf_page",
                        "fps": 0.0,
                        "content_key": key
                    })
                    # Clean up extracted page image immediately to prevent RAM disk saturation
                    if os.path.exists(img_path):
                        os.remove(img_path)

                # Accumulate rows into global buffer and flush when full
                self._accumulate_and_flush("content_key", batch_rows)

                QApplication.processEvents()
            
        except Exception as e:
            self.log_diag(f"[PDF ERROR] {e}")
        finally:
            # Cleanup temp directory
            import shutil
            try:
                if 'temp_dir' in locals() and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception:
                pass

    def process_archive_file(self, path: str):
        try:
            import zipfile
            import tarfile
            try:
                import py7zr
            except ImportError:
                py7zr = None

            self.log_diag(f"[ARCHIVE] Processing {os.path.basename(path)}...")
            
            if self.table is None:
                if not self.connect_db(): return

            active_cfg = self.model_combo.currentData()
            dim = active_cfg.get("dimension")
            m_type = active_cfg.get("type")

            # Prepare temp directory (prefer RAM disk for speed to prevent physical FS saturation)
            if os.path.exists("/dev/shm"):
                temp_dir = os.path.join("/dev/shm", f"arch_extract_{uuid.uuid4().hex}")
            else:
                temp_dir = tempfile.mkdtemp(prefix="arch_extract_")
            os.makedirs(temp_dir, exist_ok=True)

            archive_hash = self._calc_image_hash(path)
            
            # List all valid image files inside the archive
            valid_exts = ('.png', '.jpg', '.jpeg', '.tiff', '.webp')
            internal_files = []
            
            lower_path = path.lower()
            if lower_path.endswith('.zip'):
                with zipfile.ZipFile(path, 'r') as zf:
                    internal_files = [f for f in zf.namelist() if f.lower().endswith(valid_exts)]
            elif lower_path.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(path, 'r:gz') as tf:
                    internal_files = [m.name for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(valid_exts)]
            elif lower_path.endswith('.7z'):
                if not py7zr:
                    self.log_diag("[ARCHIVE ERROR] py7zr not installed. Cannot process 7z files. (pip install py7zr)")
                    return
                with py7zr.SevenZipFile(path, mode='r') as szf:
                    internal_files = [f for f in szf.getnames() if f.lower().endswith(valid_exts)]
            else:
                return

            if not internal_files:
                self.log_diag(f"[ARCHIVE] No valid images found inside {os.path.basename(path)}.")
                return

            total_files = len(internal_files)
            page_batch_size = self.archive_batch_size.value()
            
            for batch_start in range(0, total_files, page_batch_size):
                if self.stop_requested:
                    break

                # Clean temp directory before extracting new batch
                for f in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, f))
                    except Exception:
                        pass

                page_buffer_data = [] # (internal_name, text, img_path, content_key, meta_img)
                paths_to_embed_images = []

                batch_files = internal_files[batch_start:batch_start + page_batch_size]
                
                # Extract batch
                for internal_name in batch_files:
                    # Sanitize internal name for local filesystem
                    safe_internal_name = os.path.basename(internal_name)
                    img_path = os.path.join(temp_dir, f"img_{uuid.uuid4().hex}_{safe_internal_name}")
                    
                    try:
                        if lower_path.endswith('.zip'):
                            with zipfile.ZipFile(path, 'r') as zf:
                                with zf.open(internal_name) as src, open(img_path, 'wb') as dst:
                                    dst.write(src.read())
                        elif lower_path.endswith(('.tar.gz', '.tgz')):
                            with tarfile.open(path, 'r:gz') as tf:
                                member = tf.getmember(internal_name)
                                fobj = tf.extractfile(member)
                                if fobj:
                                    with open(img_path, 'wb') as dst:
                                        dst.write(fobj.read())
                        elif lower_path.endswith('.7z'):
                            with py7zr.SevenZipFile(path, mode='r') as szf:
                                szf.extract(targets=[internal_name], path=temp_dir)
                                # Rename the extracted file to our safe img_path
                                extracted_path = os.path.join(temp_dir, internal_name)
                                if os.path.exists(extracted_path):
                                    os.rename(extracted_path, img_path)
                    except Exception as extract_err:
                        self.log_diag(f"[ARCHIVE ERROR] Failed to extract {internal_name}: {extract_err}")
                        continue

                    if not os.path.exists(img_path):
                        continue

                    content_key = f"{archive_hash}::{internal_name}"
                    meta_img = json.dumps({"source": os.path.basename(path), "internal_name": internal_name, "format": "archive_slice"})
                    
                    page_buffer_data.append((internal_name, "", img_path, content_key, meta_img))
                    paths_to_embed_images.append(img_path)

                if not page_buffer_data:
                    continue

                # Check DB for existing vectors
                keys = [pd[3] for pd in page_buffer_data]
                safe_keys = [k.replace("'", "''") for k in keys]
                placeholders = ", ".join([f"'{k}'" for k in safe_keys])

                existing_matches = (
                    self.table.search()
                    .where(f"content_key IN ({placeholders})")
                    .select(["content_key", "image_vector", "text_vector"])
                    .to_arrow()
                )

                key_to_img_vec = {}
                key_to_txt_vec = {}
                if existing_matches is not None and len(existing_matches) > 0:
                    keys_col = existing_matches["content_key"].to_pylist()
                    img_vecs_col = existing_matches["image_vector"].to_pylist()
                    txt_vecs_col = existing_matches["text_vector"].to_pylist()
                    for k, iv, tv in zip(keys_col, img_vecs_col, txt_vecs_col):
                        key_to_img_vec[k] = iv
                        key_to_txt_vec[k] = tv
                del existing_matches

                # Determine what needs embedding
                imgs_to_embed = [pd[2] for pd in page_buffer_data if pd[3] not in key_to_img_vec]

                computed_img_vecs = {}
                if imgs_to_embed:
                    if m_type == "dummy" or self.active_model_obj == "dummy":
                        for p in imgs_to_embed:
                            v = np.random.rand(dim).astype(np.float32)
                            computed_img_vecs[p] = (v / np.linalg.norm(v)).tolist()
                    else:
                        import torch
                        from PIL import Image
                        with torch.no_grad():
                            self.active_model_obj.eval()
                            images = []
                            valid_paths = []
                            for p in imgs_to_embed:
                                try:
                                    img = Image.open(p)
                                    img.load()  # Force pixel decode to catch truncated/corrupted files early
                                    img = img.convert("RGB")
                                    # Fix ambiguous channel dimension warning for extremely small images (e.g., 1px width/height)
                                    if img.width < 10 or img.height < 10:
                                        img = img.resize((max(img.width, 10), max(img.height, 10)))
                                    images.append(img)
                                    valid_paths.append(p)
                                except Exception as img_err:
                                    self.log_diag(f"[ARCHIVE WARNING] Skipping corrupted/unreadable image {os.path.basename(p)}: {img_err}")
                                    if 'img' in locals():
                                        try: img.close()
                                        except Exception: pass
                            
                            if images:
                                features = self.active_model_obj.encode(images)
                                if hasattr(features, "detach"):
                                    features = features.detach().cpu().numpy()
                                features = np.array(features, dtype=np.float32)
                                norms = np.linalg.norm(features, axis=1, keepdims=True)
                                norms[norms == 0] = 1
                                features = features / norms
                                for idx, p in enumerate(valid_paths):
                                    computed_img_vecs[p] = features[idx].tolist()
                            else:
                                self.log_diag("[ARCHIVE WARNING] No valid images to encode in this batch.")

                # Construct rows and clean up images immediately
                batch_rows = []
                for internal_name, text, img_path, key, meta_img in page_buffer_data:

                        
                    # Skip images that already exist in the database
                    if key in key_to_img_vec:
                        continue
                        
                    i_vec = key_to_img_vec.get(key, computed_img_vecs.get(img_path, [0.0] * dim))
                    t_vec = [0.0] * dim # Archives primarily contain images, no text extraction yet
                    batch_rows.append({
                        "id": str(uuid.uuid4()),
                        "text_vector": t_vec,
                        "image_vector": i_vec,
                        "video_vector": [0.0] * dim,
                        "text": text,
                        "image_path": f"{path}::{internal_name}", # Virtual path for on-demand recovery
                        "metadata": meta_img,
                        "text_hash": "",
                        "image_hash": self._calc_image_hash(img_path),
                        "metadata_hash": self._calc_hash(meta_img),
                        "empty_text": True,
                        "empty_image": all(v == 0.0 for v in i_vec),
                        "empty_video": True,
                        "type": "archive_image",
                        "fps": 0.0,
                        "content_key": key
                    })
                    # Clean up extracted image immediately
                    if os.path.exists(img_path):
                        os.remove(img_path)

                # Log buffer status after embedding and buffering the batch
                self.log_diag(f"[ARCHIVE] Embedded {len(imgs_to_embed)} images. Buffer status: {len(self.ingest_buffer)}/{self.batch_buffer_size.value()}")

                # Accumulate rows into global buffer and flush when full
                self._accumulate_and_flush("content_key", batch_rows)

                QApplication.processEvents()
            
        except Exception as e:
            self.log_diag(f"[ARCHIVE ERROR] {e}")
        finally:
            # Cleanup temp directory
            import shutil
            try:
                if 'temp_dir' in locals() and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception:
                pass

    def toggle_live_monitor(self, checked: bool):
        if checked:
            directory = self.batch_dir_input.text()
            if not directory or not os.path.exists(directory):
                QMessageBox.warning(self, "Error", "Set a valid folder before enabling monitor.")
                self.chk_monitor.setChecked(False)
                return

            exts = ['.png', '.jpg', '.jpeg', '.tiff', '.webp']
            if self.chk_pdf.isChecked(): exts.append('.pdf')
            if not self.chk_assoc.isChecked():
                exts.extend(['.txt', '.json', '.py', '.c', '.h', '.html', '.md', '.js', '.cpp'])
                
            self.monitor_thread = LiveMonitorThread(directory, self.chk_recurse.isChecked(), exts)
            self.monitor_thread.files_found_signal.connect(self.process_file_batch)
            self.monitor_thread.start()
            self.log_diag(f"[MONITOR] Live tracking enabled on {directory}")
            self.batch_btn.setEnabled(False) # Disable manual while tracking
        else:
            if self.monitor_thread:
                self.monitor_thread.stop()
                self.monitor_thread = None
            self.log_diag("[MONITOR] Live tracking disabled.")
            self.batch_btn.setEnabled(True)

    
    # --- NEW CODE: Base64 and PDF Extraction Helper ---
    def _get_base64_image(self, image_path: str, metadata: dict) -> str:
        import base64
        import io
        
        # Scenario A: Image exists safely on disk
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                self.log_diag(f"[B64 ERROR] Read failure: {e}")
                return ""

        # Scenario C: Image is inside a compressed archive
        if metadata and metadata.get("format") == "archive_slice":
            internal_name = metadata.get("internal_name")
            if image_path and "::" in image_path:
                parts = image_path.split("::", 1)
                original_archive_path = parts[0]
                if os.path.exists(original_archive_path) and internal_name:
                    try:
                        import zipfile
                        import tarfile
                        try:
                            import py7zr
                        except ImportError:
                            py7zr = None
                        
                        # Use RAM disk for extraction
                        if os.path.exists("/dev/shm"):
                            temp_arch_img = os.path.join("/dev/shm", f"arch_img_{uuid.uuid4().hex}.png")
                        else:
                            temp_arch_img = tempfile.mktemp(suffix=".png")
                            
                        lower_path = original_archive_path.lower()
                        extracted = False
                        
                        if lower_path.endswith('.zip'):
                            with zipfile.ZipFile(original_archive_path, 'r') as zf:
                                with zf.open(internal_name) as src, open(temp_arch_img, 'wb') as dst:
                                    dst.write(src.read())
                            extracted = True
                        elif lower_path.endswith(('.tar.gz', '.tgz')):
                            with tarfile.open(original_archive_path, 'r:gz') as tf:
                                member = tf.getmember(internal_name)
                                fobj = tf.extractfile(member)
                                if fobj:
                                    with open(temp_arch_img, 'wb') as dst:
                                        dst.write(fobj.read())
                                    extracted = True
                        elif lower_path.endswith('.7z'):
                            if py7zr:
                                with py7zr.SevenZipFile(original_archive_path, mode='r') as szf:
                                    szf.extract(targets=[internal_name], path=os.path.dirname(temp_arch_img))
                                    # py7zr preserves internal folder structures, so we check where it went
                                    extracted_path = os.path.join(os.path.dirname(temp_arch_img), internal_name)
                                    if os.path.exists(extracted_path):
                                        os.rename(extracted_path, temp_arch_img)
                                        extracted = True
                            else:
                                self.log_diag("[B64 ERROR] py7zr missing. Cannot recover 7z archive image.")
                        
                        if extracted and os.path.exists(temp_arch_img):
                            with open(temp_arch_img, "rb") as f:
                                b64_data = base64.b64encode(f.read()).decode('utf-8')
                            os.remove(temp_arch_img)
                            self.log_diag(f"[B64 RECOVERY] Live extracted missing archive image {internal_name} to RAM disk")
                            return b64_data
                            
                    except Exception as e:
                        self.log_diag(f"[B64 ERROR] Archive live extract failed: {e}")
                        if 'temp_arch_img' in locals() and os.path.exists(temp_arch_img):
                            os.remove(temp_arch_img)

        # Scenario B: Image is missing, but it was a PDF slice. Re-extract to /dev/shm on demand.
        if metadata and metadata.get("format") == "visual_slice":
            page_num = metadata.get("page")
            if image_path and "_page_" in image_path and page_num:
                original_pdf_path = image_path.rsplit("_page_", 1)[0]
                if os.path.exists(original_pdf_path):
                    try:
                        import fitz # PyMuPDF
                        doc = fitz.open(original_pdf_path)
                        page_idx = int(page_num) - 1
                        if 0 <= page_idx < len(doc):
                            page = doc[page_idx]
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                            
                            # Re-extract to /dev/shm on demand during searches
                            if os.path.exists("/dev/shm"):
                                temp_pdf_img = os.path.join("/dev/shm", f"pdf_page_{uuid.uuid4().hex}.png")
                            else:
                                temp_pdf_img = tempfile.mktemp(suffix=".png")
                                
                            pix.save(temp_pdf_img)
                            with open(temp_pdf_img, "rb") as f:
                                b64_data = base64.b64encode(f.read()).decode('utf-8')
                            os.remove(temp_pdf_img)
                            
                            self.log_diag(f"[B64 RECOVERY] Live rendered missing page {page_num} to RAM disk from {os.path.basename(original_pdf_path)}")
                            return b64_data
                    except ImportError:
                        self.log_diag("[B64 ERROR] PyMuPDF missing. Cannot recover page.")
                    except Exception as e:
                        self.log_diag(f"[B64 ERROR] PDF live render failed: {e}")
        return ""
    # --------------------------------------------------

    def execute_search(self, query_text: str, image_path: str = "", limit: int = 3, image_base64: str = "", search_type: str = "vector") -> dict: 
        if self.table is None:
            if not self.connect_db(): return {"error": "DB not connected"}

        self.log_diag(f"[DB] Executing '{search_type.upper()}' search for: '{query_text}'")
        
        try:
            query_vector = self.do_embedding(query_text, image_path, image_base64)
            
            # Route the search based on the requested strategy
            if search_type == "fts":
                # Pure keyword search (Requires FTS index)
                results = self.table.search(query_text, query_type="fts").where("empty_text = false").with_row_id(True).limit(limit).to_list()
                
            elif search_type == "hybrid":
                # Semantic + Keyword search merged via Reciprocal Rank Fusion
                reranker = RRFReranker()
                vec_builder = self.table.search(query_vector, vector_column_name="text_vector").where("empty_text = false").with_row_id(True).limit(limit)
                fts_builder = self.table.search(query_text, query_type="fts").where("empty_text = false").with_row_id(True).limit(limit)
                # Use the explicit hybrid reranking method per API docs
                results_table = reranker.rerank_hybrid(query_text, vec_builder, fts_builder)
                results = results_table.to_pylist()

            elif search_type == "textontextvector":
                query_vector = self.do_embedding(query_text)
                vec_builder = self.table.search(query_vector, vector_column_name="text_vector").where("empty_text = false").with_row_id(True).limit(limit)
                results = vec_builder.to_list()

            elif search_type == "textonvideovector":
                query_vector = self.do_embedding(query_text)
                vec_builder = self.table.search(query_vector, vector_column_name="video_vector") \
                    .where("type = 'video_frame' AND empty_video = false").with_row_id(True).limit(limit)
                results = vec_builder.to_list()

            elif search_type == "imageonvideovector":
                query_vector = self.do_embedding(image_path=image_path, image_base64=image_base64)
                vec_builder = self.table.search(query_vector, vector_column_name="video_vector") \
                    .where("type = 'video_frame' AND empty_video = false").with_row_id(True).limit(limit)
                results = vec_builder.to_list()

            elif search_type == "textonimagevector":
                query_vector = self.do_embedding(query_text)
                vec_builder = self.table.search(query_vector, vector_column_name="image_vector").where("empty_image = false").with_row_id(True).limit(limit)
                results = vec_builder.to_list()

            elif search_type == "imageonimagevector":
                query_vector = self.do_embedding(image_path=image_path, image_base64=image_base64)
                vec_builder = self.table.search(query_vector, vector_column_name="image_vector").where("empty_image = false").with_row_id(True).limit(limit)
                results = vec_builder.to_list()
                
            else:
                # Default "vector" search -> Tri-vector RRF (Text + Image + Video)
                builder_t = self.table.search(query_vector, vector_column_name="text_vector").where("empty_text = false").with_row_id(True).limit(limit)
                builder_i = self.table.search(query_vector, vector_column_name="image_vector").where("empty_image = false").with_row_id(True).limit(limit)
                builder_v = self.table.search(query_vector, vector_column_name="video_vector").where("type = 'video_frame' AND empty_video = false").with_row_id(True).limit(limit)
                
                reranker = RRFReranker()
                results_table = reranker.rerank_multivector([builder_t, builder_i, builder_v], query=query_text)
                results = results_table.to_pylist()

            
            formatted_results = []

            for r in results:
                # --- NEW CODE: Build result and fetch base64 if enabled ---
                meta = json.loads(r["metadata"]) if r["metadata"] else {}
                result_entry = {
                    "id": r["id"],
                    "distance": r.get("_distance", 0.0),
                    "text": r["text"],
                    "image_path": r["image_path"],
                    "metadata": meta
                }
                
                if hasattr(self, 'chk_return_img_b64') and self.chk_return_img_b64.isChecked():
                    if meta.get("type") == "video_frame":
                        b64_data = self._get_video_frame_b64(r, meta)
                        if b64_data:
                            result_entry["image_base64"] = b64_data
                    elif r["image_path"] or (meta.get("format") == "visual_slice"):
                        b64_data = self._get_base64_image(r["image_path"], meta)
                        if b64_data:
                            result_entry["image_base64"] = b64_data
                
                formatted_results.append(result_entry)
                # ----------------------------------------------------------
                
            return {"status": "success", "results": formatted_results}
            
        except Exception as e:
            self.log_diag(f"[DB ERROR] {e}")
            return {"status": "error", "message": str(e)}

    @Slot()
    def manual_query(self):
        query = self.query_input.text()
        if not query: return
        
        limit = self.query_limit.value()
        strategy = self.query_strategy.currentText()
        
        results = self.execute_search(query_text=query, limit=limit, search_type=strategy)
        self.output_log.setText(json.dumps(results, indent=2))
        self.log_diag(f"[DB] Manual {strategy} search complete.")

    # =====================================================================
    # NETWORK SERVER CONTROLS
    # =====================================================================
    def toggle_server(self):
        if self.server_thread is None or not self.server_thread.isRunning():
            ip = self.ip_input.text()
            port = int(self.port_input.text())
            self.server_thread = ZmqServerThread(ip, port)
            self.server_thread.log_signal.connect(self.log_diag)
            self.server_thread.request_signal.connect(self.handle_network_request)
            self.server_thread.start()
            self.start_srv_btn.setText("Stop TCP Server")
            self.start_srv_btn.setStyleSheet("background-color: darkred; color: white;")
            
            self.connect_db()
        else:
            self.server_thread.stop()
            self.start_srv_btn.setText("Start TCP Server")
            self.start_srv_btn.setStyleSheet("")
            self.log_diag("[NETWORK] Server stopped.")

    @Slot(dict)
    def handle_network_request(self, payload: dict):
      try:
        self.log_diag(f"[AGENT] Processing remote request: {payload.get('action')}")
        action = payload.get("action")
        
        if action == "search":
            query = payload.get("query", "")
            image_path = payload.get("image_path", "") 
            image_base64 = payload.get("image_base64", "")
            limit = payload.get("limit", 3)
            search_type = payload.get("search_type", "vector") # Default fallback to vector
            
            results = self.execute_search(query, image_path, limit, image_base64, search_type)
            
            self.output_log.setText(json.dumps(results, indent=2))
            
            if self.server_thread:
                self.server_thread.send_response(results)
        elif action == "list_tables":
            self.log_diag("[AGENT] Processing remote request: list_tables")
            if self.db:
                try:
                    tables = self.db.list_tables().tables
                    res = {"status": "success", "tables": tables}
                except Exception as e:
                    res = {"status": "error", "message": f"Failed to list tables: {str(e)}"}
            else:
                res = {"status": "error", "message": "No active database connection on the agent."}
            
            if self.server_thread:
                self.server_thread.send_response(res)

        elif action == "load_table":
            self.log_diag("[AGENT] Processing remote request: load_table")
            table_name = payload.get("db_name") or payload.get("table_name")
            
            if not self.db:
                res = {"status": "error", "message": "No active database connection on the agent."}
            elif not table_name:
                res = {"status": "error", "message": "Missing 'table_name' parameter in request."}
            else:
                try:
                    existing_tables = self.db.list_tables().tables
                    if table_name in existing_tables:
                        # Safely open the existing table without triggering a creation event
                        self.table = self.db.open_table(table_name)

                        # Retrieve the list of all constructed indices
                        indices = self.table.list_indices()
                        for tindex in indices:
                            print(f"Index Name: {tindex.name}")
                            print(f"Type: {tindex.index_type}")
                            print(f"Columns: {tindex.columns}")
    
                            # If using newer LanceDB core versions, extended metadata can be accessed:
                            if hasattr(tindex, "index_details"):
                                print(f"Details: {tindex.index_details}")
                        
                        # Synchronize the GUI state to reflect the network action
                        self.table_input.blockSignals(True)
                        self.table_combo.blockSignals(True)
                        
                        self.table_input.setText(table_name)
                        if self.table_combo.findText(table_name) == -1:
                            self.table_combo.addItem(table_name)
                        self.table_combo.setCurrentText(table_name)
                        
                        self.table_input.blockSignals(False)
                        self.table_combo.blockSignals(False)
                        
                        self.log_diag(f"[TABLE Successfully loaded existing table '{table_name}'.")
                        res = {"status": "success", "message": f"Table '{table_name}' loaded successfully."}

                        if self.server_thread:
                             self.server_thread.send_response({"status": "success", "message": f"Loaded '{table_name}'"})

                    else:
                        res = {"status": "error", "message": f"Table '{table_name}' does not exist in the current database."}
                except Exception as e:
                    self.log_diag(f"[TABLE ERROR] Remote load failed: {e}")
                    res = {"status": "error", "message": f"Failed to load table: {str(e)}"}
            
            if self.server_thread:
                self.server_thread.send_response(res)

        
        elif action == "ping":
            self.output_log.setText("Ping received.")
            if self.server_thread:
                self.server_thread.send_response({"status": "alive", "agent": "rag_lancedb", "active_model": self.model_combo.currentText()})
                
        else:
            err = {"status": "error", "message": "Unknown action"}
            self.output_log.setText(json.dumps(err))
            if self.server_thread:
                self.server_thread.send_response(err)
      except Exception as e:
        self.log_diag(f"[NETWORK ERROR] {str(e)}")
        # Critical: Always send a response to unlock the ZmqServerThread
        if self.server_thread:
            self.server_thread.send_response({"status": "error", "message": str(e)})

    def browse_video_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Video Source Directory")
        if dir_path:
            self.video_dir_input.setText(dir_path)

    def run_video_ingestion(self):
        directory = self.video_dir_input.text()
        if not directory or not os.path.exists(directory):
            QMessageBox.warning(self, "Error", "Invalid directory selected.")
            return

        fps = self.video_fps.value()
        exts = ['.mp4', '.mkv', '.avi']
        
        video_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in exts:
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            QMessageBox.warning(self, "Info", "No video files found.")
            return

        self.set_gui_enabled(False)
        self.video_ingest_btn.setEnabled(False)
        
        try:
            active_cfg = self.model_combo.currentData()
            dim = active_cfg.get("dimension", 512)
            m_type = active_cfg.get("type")
            buffer_limit = self.batch_buffer_size.value() # NEW: Use GUI config

            # Prepare temp directory (prefer RAM disk for speed)
            if os.path.exists("/dev/shm"):
                temp_base = "/dev/shm/sentry_extract"
            else:
                temp_base = tempfile.mkdtemp(prefix="sentry_extract_")
            os.makedirs(temp_base, exist_ok=True)

            # Phase 1 & 2 merged: Process each video file sequentially
            # NEW: Decouple buffer from video loop. Accumulates across ALL videos.
            row_buffer = [] 
            
            total_videos = len(video_files)
            self.video_progress.setMaximum(total_videos)
            
            for v_idx, video_path in enumerate(video_files):
                if self.stop_requested:
                    self.log_diag("[SYSTEM] Stop requested. Aborting video ingestion.")
                    break
                    
                current_video_idx = v_idx + 1
                self.log_diag(f"[VIDEO] Processing video {current_video_idx}/{total_videos}: {os.path.basename(video_path)}")
                
                video_dir = os.path.join(temp_base, f"vid_{v_idx}")
                os.makedirs(video_dir, exist_ok=True)
                
                # --- SMART VIDEO SAMPLING SKIP ---
                if self.chk_smart_video_skip.isChecked():
                    native_fps, duration = self._probe_video(video_path)
                    if duration > 0:
                        sampling_fps = fps
                        timestamps = []
                        for i in range(4):
                            t = duration - (i / sampling_fps)
                            if t < 0: t = 0.0
                            timestamps.append(t)
                        timestamps = sorted(list(set(timestamps)))
                        
                        probe_keys = []
                        probe_files = []
                        for t in timestamps:
                            frame_idx = max(1, int(round(t * sampling_fps)))
                            frame_file = f"frame_{frame_idx:08d}.jpg"
                            content_key = f"{video_path}_{fps}_{frame_file}"
                            probe_keys.append(content_key)
                            
                            temp_frame_path = os.path.join(video_dir, f"probe_{frame_idx:08d}.jpg")
                            probe_files.append(temp_frame_path)
                        
                            cmd_probe = [
                                "ffmpeg", "-threads", "4", "-filter_threads", "4", "-nostdin", "-ss", str(t), "-i", video_path,
                                "-frames:v", "1", "-q:v", "2", temp_frame_path, "-y"
                            ]
                            subprocess.run(cmd_probe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            
                        safe_keys = [k.replace("'", "''") for k in probe_keys]
                        placeholders = ", ".join([f"'{k}'" for k in safe_keys])
                        
                        existing_matches = (
                            self.table.search()
                            .where(f"content_key IN ({placeholders})")
                            .select(["content_key"])
                            .to_arrow()
                        )
                    
                        existing_keys = set(existing_matches["content_key"].to_pylist()) if existing_matches is not None and len(existing_matches) > 0 else set()
                    
                        for pf in probe_files:
                            if os.path.exists(pf):
                                os.remove(pf)
                            
                        if len(existing_keys) == len(probe_keys):
                            self.log_diag(f"[VIDEO] Smart Skip: All {len(probe_keys)} probe frames found in DB. Skipping {os.path.basename(video_path)}.")
                            self.video_progress.setValue(current_video_idx)
                            # Clean up the empty video directory before continuing
                            if os.path.exists(video_dir):
                                try:
                                    os.rmdir(video_dir)
                                except Exception:
                                    pass
                            QApplication.processEvents()
                            continue
                        else:
                            self.log_diag(f"[VIDEO] Smart Skip: Probe incomplete. Full extraction for {os.path.basename(video_path)}.")
                # -----------------------------------

                            # Extract frames for this single video in batches to prevent disk saturation
                            batch_frame_limit = 4096
                            batch_duration = batch_frame_limit / fps
                            current_time = 0.0
                            total_extracted_frames = 0
            
                            while True:
                                if self.stop_requested:
                                    break
                    
                                # Clean directory before each batch extraction
                                for f in os.listdir(video_dir):
                                    try:
                                        os.remove(os.path.join(video_dir, f))
                                    except Exception:
                                        pass
                                        
                                start_frame_number = int(round(current_time * fps)) + 1
                                
                                cmd = [
                                    "ffmpeg", "-threads", "4", "-filter_threads", "4", "-nostdin",
                                    "-ss", str(current_time),
                                    "-i", video_path,
                                    "-vf", f"fps={fps}", "-q:v", "2",
                                    "-frames:v", str(batch_frame_limit),
                                    "-start_number", str(start_frame_number),
                                    f"{video_dir}/frame_%08d.jpg", "-y"
                                ]
                
                                devnull = open(os.devnull, "w")
                                try:
                                    subprocess.run(cmd, stdout=devnull, stderr=devnull, check=False)
                                finally:
                                    devnull.close()
                
                                frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
                                num_frames = len(frames)
                                
                                if num_frames == 0:
                                    break
                    
                                total_extracted_frames += num_frames
                                self.log_diag(f"[VIDEO] Extracted batch of {num_frames} frames (starting at {start_frame_number}) from {os.path.basename(video_path)}.")
                
                                # Process frames for this batch in chunks (for embedding efficiency)
                                chunk_size = self.video_batch_size.value()
                                for i in range(0, num_frames, chunk_size):
                                    if self.stop_requested:
                                        break
                                    chunk_frames = frames[i : i + chunk_size]
                    
                                    chunk_paths = []
                                    chunk_keys = []
                                    chunk_frame_files = []
                                    
                                    for f_file in chunk_frames:
                                        frame_path = os.path.join(video_dir, f_file)
                                        content_key = f"{video_path}_{fps}_{f_file}"
                                        chunk_paths.append(frame_path)
                                        chunk_keys.append(content_key)
                                        chunk_frame_files.append(f_file)
                
                                    # Check DB for existing vectors in this chunk
                                    safe_keys = [k.replace("'", "''") for k in chunk_keys]
                                    placeholders = ", ".join([f"'{k}'" for k in safe_keys])
                
                                    # --- FIX: use Arrow directly, skip the pandas round-trip ---
                                    existing_matches = (
                                        self.table.search()
                                        .where(f"content_key IN ({placeholders})")
                                        .select(["content_key", "video_vector"])
                                        .to_arrow()
                                    )

                                    key_to_vec = {}
                                    if existing_matches is not None and len(existing_matches) > 0:
                                        keys_col = existing_matches["content_key"].to_pylist()
                                        vecs_col = existing_matches["video_vector"].to_pylist()
                                        key_to_vec = dict(zip(keys_col, vecs_col))
                                    del existing_matches

                                    # Identify frames needing fresh embedding
                                    paths_to_embed = []
                                    for idx, key in enumerate(chunk_keys):
                                        if key not in key_to_vec:
                                            paths_to_embed.append(chunk_paths[idx])
                                                
                                    # Batch Embedding
                                    computed_vecs = {}
                                    if paths_to_embed:
                                        if m_type == "dummy" or self.active_model_obj == "dummy":
                                            for p in paths_to_embed:
                                                v = np.random.rand(dim).astype(np.float32)
                                                computed_vecs[p] = (v / np.linalg.norm(v)).tolist()
                                        else:
                                            import torch
                                            from PIL import Image
                                            with torch.no_grad():
                                                self.active_model_obj.eval()
                
                                                # --- FIX: Explicitly load images, encode them, THEN close ---
                                                images = []
                                                valid_paths = []  # Track which paths successfully loaded
                                                try:
                                                    for p in paths_to_embed:
                                                        try:
                                                            img = Image.open(p)
                                                            img.load()                       # force pixel decode while FD is open
                                                            images.append(img.convert("RGB"))
                                                            valid_paths.append(p)
                                                        except Exception as img_err:
                                                            # Skip corrupted/truncated frames gracefully
                                                            self.log_diag(f"[VIDEO WARNING] Skipping corrupted frame {os.path.basename(p)}: {img_err}")
                                                            if 'img' in locals():
                                                                try: img.close()
                                                                except Exception: pass
                            
                                                    # Encode while images are still open
                                                    if images:
                                                        features = self.active_model_obj.encode(images)
                                                    else:
                                                        features = np.array([])  # No valid images to encode 

                                                finally:
                                                    # Close images AFTER encoding is complete
                                                    for img in images:
                                                        try: img.close()
                                                        except Exception: pass
                
                                                # Drop image references so their buffers can be freed
                                                del images
                
                                                if len(features) > 0:
                                                    if hasattr(features, "detach"):
                                                        features = features.detach().cpu().numpy()
                                                    features = np.array(features, dtype=np.float32)
                                                    norms = np.linalg.norm(features, axis=1, keepdims=True)
                                                    norms[norms == 0] = 1
                                                    features = features / norms
                                                    for idx, p in enumerate(valid_paths):
                                                        computed_vecs[p] = features[idx].tolist()

                                                # Release CUDA/CPU allocator caches periodically
                                                del features
                                                if torch.cuda.is_available():
                                                    torch.cuda.empty_cache()

                                    # Construct rows for upsert -> ONLY BUFFER NEW RECORDS
                                    for idx, (key, path, f_file) in enumerate(zip(chunk_keys, chunk_paths, chunk_frame_files)):
                                        if key in key_to_vec:
                                            continue # Already in DB, skip to avoid merge_insert overhead and maintain deduplication
                            
                                        v_vec = computed_vecs[path]
                                        meta = json.dumps({"source": os.path.basename(video_path), "frame_file": f_file, "type": "video_frame", "source_video": video_path, "fps": fps})
                                        row_buffer.append({
                                            "id": str(uuid.uuid4()),
                                            "text_vector": [0.0] * dim,
                                            "image_vector": [0.0] * dim,
                                            "video_vector": v_vec,
                                            "text": "",
                                            "image_path": "",
                                            "metadata": meta,
                                            "text_hash": "",
                                            "image_hash": "",
                                            "metadata_hash": self._calc_hash(meta),
                                            "empty_text": True,
                                            "empty_image": True,
                                            "empty_video": False,
                                            "type": "video_frame",
                                            "fps": fps,
                                            "content_key": key
                                        })

                                    # Cleanup temp frames for this chunk immediately
                                    for path in chunk_paths:
                                        if os.path.exists(path):
                                            os.remove(path)

                                    ### diagnostics on terminal
                                    print(f"video row_buffer status: {len(row_buffer)} / {buffer_limit}  ")
                
                                    # Trigger atomic append ONLY when threshold is reached
                                    if len(row_buffer) >= buffer_limit:
                                        self.table.add(row_buffer) # Use add() instead of merge_insert() for massive memory savings
                                        row_buffer.clear()
                                        gc.collect() # Force GC to reclaim memory from cleared buffer
                                        QApplication.processEvents()

                                    QApplication.processEvents()
                    
                                # Advance time for the next batch
                                current_time += batch_duration

                            # Cleanup temp directory for this video after ALL batches are processed
                            if os.path.exists(video_dir):
                                import shutil
                                try:
                                    shutil.rmtree(video_dir)
                                except Exception as cleanup_err:
                                    self.log_diag(f"[VIDEO WARNING] Failed to cleanup {video_dir}: {cleanup_err}")
                
                # Update progress (per video)
                self.video_progress.setValue(current_video_idx)
                self._log_memory(f"Video {current_video_idx}/{total_videos} DONE")
                QApplication.processEvents()
                self.log_diag(f"[VIDEO] Finished ingesting {os.path.basename(video_path)}.")


            ## --- CRITICAL FIX: Perform compaction and cleanup ONCE at the end ---
            ## Doing this inside the flush loop causes massive memory bloat and fragmentation
            #self.log_diag("[DB] Ingestion finished. Compacting files and cleaning up old versions...")

            # NEW: Flush remaining records after ALL videos are processed
            if row_buffer:
                self.table.add(row_buffer)
                row_buffer.clear()
                gc.collect()
                QApplication.processEvents()
            
            # Centralized end-of-ingestion maintenance
            self._finalize_ingestion()
            #self.log_diag("[DB] Ingestion finished. Compacting files and cleaning up old versions...")

            #try:
            #    self.table.compact_files()
            #    import datetime
            #    self.table.cleanup_old_versions(older_than=datetime.timedelta(minutes=0), delete_unverified=True)
            #    self.log_diag("[DB] Compaction and cleanup complete.")
            #except Exception as e:
            #    self.log_diag(f"[DB ERROR] Compaction failed: {e}")
    
            # --- FIX: cap the processed-files set so it can't grow without bound ---
            if len(self.processed_files) > 50_000:
                self.processed_files.clear()
                self.log_diag("[MEM] Cleared processed_files set to cap memory usage.")
            
            self.log_diag(f"[VIDEO] Ingestion complete. Processed {total_videos} video files.")

            
        except Exception as e:
            self.log_diag(f"[VIDEO ERROR] {e}")
        finally:
            self.set_gui_enabled(True)
            self.video_ingest_btn.setEnabled(True)


    def _get_video_frame_b64(self, record: dict, metadata: dict) -> str:
        """Reconstructs a specific frame from the source video on-demand using FFmpeg seeking."""
        import base64
        video_path = metadata.get("source_video")
        if not video_path or not os.path.exists(video_path):
            return ""
        
        frame_file = metadata.get("frame_file")
        fps = record.get("fps", 1.0)
        
        match = re.search(r'frame_(\d+)\.jpg', frame_file)
        if not match: return ""
        
        frame_idx = int(match.group(1))
        timestamp = frame_idx / fps
        
        if os.path.exists("/dev/shm"):
            temp_png = os.path.join("/dev/shm", f"vid_frame_{uuid.uuid4().hex}.png")
        else:
            temp_png = tempfile.mktemp(suffix=".png")
            
        cmd = [
            "ffmpeg", "-threads", "4", "-filter_threads", "4", "-nostdin", "-ss", str(timestamp), "-i", video_path,
            "-frames:v", "1", "-q:v", "2", temp_png, "-y"
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(temp_png) and os.path.getsize(temp_png) > 0:
                with open(temp_png, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                return b64
        except Exception as e:
            self.log_diag(f"[VIDEO B64 ERROR] {e}")
        finally:
            if os.path.exists(temp_png): os.remove(temp_png)
        return ""

    def closeEvent(self, event):
        if self.server_thread and self.server_thread.isRunning():
            self.server_thread.stop()
        if self.monitor_thread and self.monitor_thread.isRunning():
            self.monitor_thread.stop()
        event.accept()

# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = RagAgentWindow()
    window.show()
    sys.exit(app.exec())


