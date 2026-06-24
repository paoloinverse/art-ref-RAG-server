# Art-ref-RAG-server
CLIP-ViT-L-14 + LanceDB based RAG server for large-scale storage of artistic references, offering FTS, booru tag, natural language and image/sketch vector search. Query with the Art-ref-RAG-frontend


# Multimodal RAG Agent for Artists — Drawing Reference Server

A general-purpose, hardware-friendly **Multimodal Retrieval-Augmented Generation (RAG) server** built originally for storing and searching drawing references for artists. It can ingest images (with optional companion textual metadata), PDFs (auto-converted to text + page images), CSV files, and videos (with frame sampling), and exposes a ZeroMQ TCP socket so any front-end application can perform semantic, full-text, hybrid, and cross-modal searches.

Results can be returned together with the actual image content pulled from local storage (or reconstructed on-demand for video frames and PDF page slices), making it ideal for building reference browsers, mood-board tools, or AI-assisted illustration assistants.

---

## Table of Contents

- [Key Features](#key-features)
- [Supported Search Types](#supported-search-types)
- [Hardware & Thermal Optimization Philosophy](#hardware--thermal-optimization-philosophy)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Graphical User Interface Reference](#graphical-user-interface-reference)
- [Available CLIP-ViT Models](#available-clip-vit-models)
- [Network / Front-end Integration](#network--front-end-integration)
- [Recommended Settings](#recommended-settings)
- [Screenshots](#screenshots)
- [License](#license)

---

## Key Features

- **Multi-format ingestion**: PNG, JPG, JPEG, TIFF, WebP (auto-converted to PNG), PDF (PyMuPDF), CSV, plain text/code files, and videos (MP4, MKV, AVI).
- **Companion metadata auto-association**: any `.txt` or `.json` file whose basename matches an image is automatically attached as searchable content.
- **Tri-modal vector storage**: every record keeps separate `text_vector`, `image_vector`, and `video_vector` columns so you can query each modality independently.
- **Six search modes** plus a default tri-vector hybrid fusion via Reciprocal Rank Fusion (RRF).
- **ZeroMQ REP socket** for connecting external front-ends (ComfyUI custom nodes, web dashboards, Blender add-ons, etc.).
- **On-demand image delivery**: the server can return matching images embedded as Base64 directly in the JSON reply, including PDF page slices and video frames (re-extracted via FFmpeg seeking).
- **Live directory monitoring**: drop new files into a watched folder and the agent will periodically scan and ingest them.
- **Hardware-aware design**: decoupled buffering, hash-based deduplication, batched embedding, RAM-disk frame extraction, and end-of-batch LanceDB compaction all minimize SSD wear, thermal spikes, and power consumption.

---

## Supported Search Types

| Search Type | Query Modality | Target Vector Column | Description |
|---|---|---|---|
| `vector` (default) | Text or image | text + image + video (RRF fusion) | Tri-vector Reciprocal Rank Fusion across all modalities. |
| `fts` | Text (keywords) | — | Pure Full-Text Search on the `text` column. Requires building the FTS index. Recommended for text searches only |
| `hybrid` | Text | text_vector + FTS (RRF) | Combines semantic and keyword retrieval. |
| `textontextvector` | Text | `text_vector` | Semantic text-to-text similarity. Recommended |
| `textonimagevector` | Text | `image_vector` | Describe an image in words and find matching visuals. Recommended |
| `imageonimagevector` | Image (or sketch) | `image_vector` | Search by reference image or even a rough sketch. Recommended |
| `textonvideovector` | Text | `video_vector` | Find video frames matching a textual description. Recommended |
| `imageonvideovector` | Image | `video_vector` | Find video frames visually similar to a reference image. Recommended |

---

## Hardware & Thermal Optimization Philosophy

This system was conceived to run for long sessions on modest hardware (including laptops) without overheating the GPU, hammering the SSD, or ballooning RAM. Of course it will shine on gaming PCs or more powerful hardware and is able to ingest / search through millions of images in a single table. The ingestion pipeline is built around the following principles:

1. **Decoupled embedding and writing.** Embeddings are produced in configurable batches and accumulated into an in-memory row buffer. LanceDB writes are performed only when the buffer reaches the configured `Record Buffer Limit`, replacing the much heavier `merge_insert` path with simple `add()` appends. This avoids the O(N²) write amplification that occurs when every record triggers a merge operation.

2. **Hash-based deduplication cache.** Before invoking the neural model, the agent computes SHA-256 hashes of the text, the image bytes, and the metadata. If a hash already exists in the database, the previously stored vector is reused, so the GPU is never asked to re-encode content it has already seen. This is the single largest source of thermal and power savings during repeated ingestion runs.

3. **Batched inference.** Text, image, PDF page, and video frame embeddings are processed in chunks (configurable per modality). This keeps GPU VRAM usage predictable and prevents thermal spikes caused by large one-shot batches.

4. **Centralized maintenance.** File compaction (`compact_files`) and old-version cleanup (`cleanup_old_versions`) are deferred to a single call at the end of an ingestion cycle (`_finalize_ingestion`). Running them inside the hot loop would cause quadratic SSD writes and RSS inflation.

5. **PyArrow allocator tuning.** At startup the agent tells mimalloc/jemalloc to return freed pages to the OS within 10 ms (`pa.mimalloc_set_decay_ms(10)` / `pa.jemalloc_set_decay_ms(10)`). Without this, the allocator would keep freed pages in its internal pool forever, making resident set size appear to grow monotonically.

6. **Explicit garbage collection.** `gc.collect()` is invoked after each buffer flush and after each video batch, and CUDA caches are emptied between batches when a GPU is available.

7. **RAM-disk frame extraction.** When available, video frames are extracted to `/dev/shm` (a tmpfs in RAM) instead of the SSD, eliminating write amplification during video ingestion. FFmpeg's verbose output is discarded to avoid buffering megabytes of stderr in memory.

8. **Image handle hygiene.** Pillow images are explicitly `load()`-ed and `close()`-d after encoding, and references are dropped so their decode buffers can be reclaimed immediately.

9. **GUI locking during ingestion.** Interactive controls are disabled while a synchronous ingestion is in progress, preventing out-of-band requests that could corrupt the database.

---

## Requirements

### System-level dependencies

- **Python 3.10+**
- **FFmpeg** — required for video ingestion and on-demand video frame reconstruction. Must be on your `PATH`.
- A CUDA-capable GPU is **optional** but strongly recommended for image and video ingestion. The agent will automatically fall back to CPU if no GPU is detected.

### Python packages

- `PySide6` — graphical user interface
- `pyzmq` — ZeroMQ TCP server
- `lancedb` — vector database
- `pyarrow` — columnar data layer used by LanceDB
- `numpy`
- `pillow`
- `torch` and `torchvision`
- `sentence-transformers`
- `transformers`
- `huggingface_hub`
- `pymupdf` — PDF extraction (only required for PDF ingestion)
- `psutil` — memory diagnostics
- `gc` — garbage collector

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/paoloinverse/art-ref-RAG-server.git
cd art-ref-RAG-server
```

### 2. Create and activate a virtual environment

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (cmd):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Upgrade pip and install the dependencies

```bash
python -m pip install --upgrade pip
pip install PySide6 pyzmq lancedb pyarrow numpy pillow
pip install torch torchvision
pip install sentence-transformers transformers huggingface_hub
pip install pymupdf psutil gc
```

> **Tip — CUDA-enabled PyTorch:**  
> If you have an NVIDIA GPU, install the CUDA build of PyTorch that matches your driver. For example, for CUDA 12.1:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Ensure FFmpeg is installed

**Debian/Ubuntu:**
```bash
sudo apt install ffmpeg
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Windows:** download a build from <https://www.gyan.dev/ffmpeg/builds/> and add its `bin` folder to your `PATH`.

---

## Running the Application

From the repository root, with the virtual environment activated:

```bash
python rag_agent_gui12z2_clip-vit_fast.py
```

The GUI will appear with the title **"Multimodal RAG Agent (General Purpose)"**. By default the agent starts with the **Dummy** model active so you can test the full ingestion and search pipeline without downloading any weights.

---

## Graphical User Interface Reference

### Master Configuration (top panel)

| Element | Function |
|---|---|
| **Bind IP / TCP Port** | Network address the ZeroMQ REP socket listens on. Defaults to `127.0.0.1:5001`. Use `0.0.0.0` to expose the agent to other machines on your LAN. |
| **Work Directory** | Folder where LanceDB stores its `.lance` tables. Defaults to `./lancedb_data`. |
| **Available DBs** | Recursive scanner that locates every `.lance` table under the work directory. Selecting one reconnects the agent to it. |
| **Scan** | Re-runs the recursive database scan. |
| **New Table Name / Create New** | Creates a new LanceDB table with the multimodal schema (vectors, hashes, empty-flags, content key, etc.). |
| **Available Tables / Scan Tables / Load Selected** | Lists tables in the current work directory and loads the selected one. |
| **Select Model / Load / Download Selected Model** | Opens the model picker and triggers a background download + load via `ModelLoaderThread`. The GUI stays responsive during the operation. |
| **Save Config / Load Config** | Persists the current configuration (IP, port, work dir, table name, batch sizes, model index) to a JSON file. |
| **Start TCP Server** | Binds the ZeroMQ REP socket and begins listening for front-end requests. Toggles to **Stop TCP Server** while running. |
| **Return Local Images as Base64 in Search** | When checked, search responses include the matching image (or reconstructed PDF/video frame) encoded as Base64, so the front-end doesn't need direct filesystem access. |

### Ingestion Tabs

#### Single Item

Manual one-shot ingestion. Provide a text description, an image path (PNG/TIFF/JPG), and optional metadata as a JSON string, then click **Embed and Add to LanceDB**.

#### Batch / Auto

Bulk ingestion from a folder.

| Element | Function |
|---|---|
| **Source Folder** | Directory to scan. |
| **Auto-associate matching .txt and .json files** | When enabled, images whose basename matches a `.txt` or `.json` sibling will be ingested as text+image pairs (routed through the single-item pipeline). |
| **Recursive (Scan Subfolders)** | Walks subdirectories. |
| **Process PDFs (Requires PyMuPDF)** | Enables PDF ingestion. Each page is rasterized to PNG and its extracted text is stored alongside. |
| **Scan and ingest CSV files** | Each CSV row becomes a searchable record. |
| **Use first CSV row as metadata fields** | Treats the CSV header as field names and stores typed values (int/float/string) in the metadata JSON. |
| **CSV Batch Window** | Number of CSV rows to embed and flush together (hard-capped at 500 due to LanceDB's WHERE-condition limit). |
| **Record Buffer Limit** | Global buffer threshold (in rows) before a LanceDB append is triggered. Higher = fewer writes but more RAM. |
| **Text Embedding Batch / Image Embedding Batch / PDF Page Embedding Batch / Video Frame Embedding Batch** | Per-modality inference batch sizes. |
| **Enable Live Periodic Monitoring** | Spawns a background thread that re-scans the source folder every 10 seconds and ingests any new files. |
| **Run Manual Batch Ingest Now** | Triggers a one-shot batch ingestion. |

#### Video Pipeline

| Element | Function |
|---|---|
| **Source Folder** | Folder containing `.mp4`, `.mkv`, or `.avi` files. |
| **Extraction FPS** | Frame sampling rate (0.1–30.0). `1.0` = one frame per second. |
| **Progress bar** | Per-video progress indicator. |
| **Extract Frames & Embed to LanceDB** | Runs FFmpeg to extract frames (to `/dev/shm` when available), embeds them in batches, and appends them to the active table with `type = "video_frame"`. |

### Manual Debug Query

| Element | Function |
|---|---|
| **Vector Prompt** | The query text or, for image searches, the path to a reference image/sketch. |
| **Search Strategy** | `vector` (tri-vector RRF), `fts`, or `hybrid`. The full set of search types (`textontextvector`, `textonimagevector`, `imageonimagevector`, `textonvideovector`, `imageonvideovector`) is exposed through the network API via the `search_type` field. |
| **Max Results** | Number of results to return (1–256). |
| **Rebuild FTS Index** | Creates/replaces the Full-Text Search index on the `text` column. Required before `fts` or `hybrid` searches work. |
| **Build Vector Index (Post-Ingestion)** | Builds an IVF_HNSW_SQ cosine index on `video_vector` with 256 partitions. Run this once after bulk ingestion to accelerate ANN queries. |
| **Search Database** | Executes the query and prints the JSON result in the **Live Output Data** panel. |

### Diagnostic Log & Live Output Data

- **Diagnostic Log** (lower-right, top): verbose trace of model loading, database operations, deduplication hits, memory snapshots, and network events. Auto-trims itself after 10,000 lines to prevent unbounded memory growth.
- **Live Output Data** (lower-right, bottom): the raw JSON returned by the last search or network request. Includes a **Copy Output to Clipboard** button.

---

## Available CLIP-ViT Models

The model registry is defined in `RagAgentWindow.__init__`. Models are loaded in a background `QThread` so the GUI never freezes during download or initialization. Hardware is auto-detected: CUDA is used if available, otherwise CPU.

| Display Name | HuggingFace Repo | Vector Dim | Notes |
|---|---|---|---|
| **Dummy (Instant Start/Debug)** | `dummy` | 512 | Generates normalized random vectors. Use for testing the pipeline without downloading any weights. |
| **CLIP ViT-B-32 (Stable OpenCLIP)** | `sentence-transformers/clip-ViT-B-32` | 512 | The stable, battle-tested OpenCLIP B/32. Fast and lightweight — the best starting point. |
| **CLIP ViT-B-32 multilingual-v1** | `sentence-transformers/clip-ViT-B-32-multilingual-v1` | 512 | Multilingual text encoder. **Recommended when your metadata or queries are in Japanese** (or other non-English languages). Same vector dimension as the English B/32, so it is fully interchangeable for text-vector operations. |
| **CLIP ViT-L-14** | `sentence-transformers/clip-ViT-L-14` | 768 | Larger model with substantially higher accuracy. Because the vector dimension differs (768 vs 512), it **cannot share tables** with B/32 models — create a new table after loading (vectors default to 512 dimension if this model is not loaded first). |

> **Important:** Changing the active model after a table has been created with a different dimension will cause a schema mismatch. The agent calls `connect_db()` automatically after a model load to validate this; create or load a different table if you switch between B/32-family and L/14 models.

---

## Network / Front-end Integration

The agent exposes a ZeroMQ REP socket on the configured IP/port. Front-ends send JSON requests and receive JSON replies. Supported actions:

### `ping`
```json
{ "action": "ping" }
```
Reply: `{"status": "alive", "agent": "rag_lancedb", "active_model": "..."}`

### `search`
```json
{
  "action": "search",
  "query": "a red car at sunset",
  "image_path": "",
  "image_base64": "",
  "limit": 5,
  "search_type": "vector"
}
```
Valid `search_type` values: `vector`, `fts`, `hybrid`, `textontextvector`, `textonimagevector`, `imageonimagevector`, `textonvideovector`, `imageonvideovector`.

If **Return Local Images as Base64 in Search** is enabled, each result includes an `image_base64` field. For video frames, the agent re-extracts the exact frame from the source video via FFmpeg seeking — no frame images are persisted on disk.

### `list_tables`
```json
{ "action": "list_tables" }
```
Reply: `{"status": "success", "tables": ["general_collection", "illustrations_v2", ...]}`

### `load_table`
```json
{ "action": "load_table", "table_name": "illustrations_v2" }
```
Switches the active table from a remote request and synchronizes the GUI.

---

## Recommended Settings

These values work well on a typical developer laptop (16 GB RAM, mid-range NVIDIA GPU, NVMe SSD). Adjust based on your hardware.

### General

- **Bind IP:** `127.0.0.1` (local front-end) or `0.0.0.0` (LAN-accessible).
- **TCP Port:** `5001`.
- **Return Local Images as Base64:** enable only if your front-end cannot read the local filesystem. Adds significant payload size to each reply.

### Model selection

- **First-time / testing:** Dummy model.
- **English-only metadata, fast operation:** `CLIP ViT-B-32 (Stable OpenCLIP)`.
- **Japanese or multilingual metadata:** `CLIP ViT-B-32 multilingual-v1`.
- **Maximum accuracy, dedicated table:** `CLIP ViT-L-14` (remember the vector dimension is 768, when creating new tables load the model first).

### Batch / Auto ingestion

| Parameter | Recommended | Reasoning |
|---|---|---|
| Record Buffer Limit | Default `5000`–`10000` | Balances write frequency against RAM usage. Lower it on machines with ≤8 GB RAM. I routinely use 100000 to minimize NVME wear on large ingestions, with 32GB RAM |
| Text Embedding Batch | `64` | Good throughput on GPU; safe on CPU. 1024 is perfectly safe to use. |
| Image Embedding Batch | `64` | Fits comfortably in 6–8 GB VRAM with L/14. Anything above `64` tends to linearly increase the processing time, I routinely use 256 on my system. |
| PDF Page Embedding Batch | `32` | Pages are 2× rasterized PNGs and consume more memory. |
| Video Frame Embedding Batch | `32` | Keeps VRAM usage stable during long video runs. Anything above `64` tends to linearly increase the processing time, I routinely use 256 on my system. |
| CSV Batch Window | `256` | Stays well under LanceDB's 500-condition WHERE limit. |

### Video Pipeline

- **Extraction FPS:** `1.0` for general reference libraries; `0.5` for long videos; `2.0` only for short clips where motion detail matters.
- Prefer running on Linux with `/dev/shm` available — frame extraction then happens entirely in RAM.

### Post-ingestion maintenance

- After any large bulk ingestion, click **Build Vector Index (Post-Ingestion)** once to construct the IVF_HNSW_SQ index on `video_vector`.
- Click **Rebuild FTS Index** whenever you ingest a significant amount of new text content and intend to use `fts` or `hybrid` search.

### Memory-conscious long sessions

- If you expect to ingest hundreds of thousands of files, lower the Record Buffer Limit to `5000`–`10000` so each flush reclaims memory faster.
- The diagnostic log auto-trims after 10,000 lines; no manual maintenance is needed.
- On Linux, monitor RSS with `htop`; on Windows, watch the process in Task Manager. A slow upward drift during ingestion is normal and will settle after `_finalize_ingestion` runs.

---

## Screenshots

> _This section will be populated with screenshot samples of the GUI in action._
>
> <!--
> Recommended screenshots to add:
> 1. Main window with Master Configuration panel visible.
> 2. Batch / Auto tab during a live ingestion run.
> 3. Video Pipeline tab with the progress bar mid-run.
> 4. Manual Debug Query returning hybrid search results.
> 5. Diagnostic Log showing deduplication cache hits and memory snapshots.
> 6. Example front-end integration consuming the ZeroMQ API.
>
> Example markdown once images are added:
> ![Main Window](screenshots/main_window.png)
> ![Batch Ingestion](screenshots/batch_ingestion.png)
> ![Video Pipeline](screenshots/video_pipeline.png)
> ![Hybrid Search Results](screenshots/hybrid_search.png)
> -->

---

## License

Apache-2.0
