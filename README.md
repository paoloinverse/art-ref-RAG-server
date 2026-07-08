# Artistic References RAG server
**Multimodal RAG Agent/Server for Artists — Drawing References Search Server**

A general-purpose Multimodal Retrieval-Augmented Generation (RAG) server built for storing and searching artistic references. It ingests images, PDFs, videos, and text, and exposes a ZeroMQ TCP API so any front-end application can perform semantic, full-text, and cross-modal searches. 

Designed to run efficiently on modest hardware (including laptops) without overheating, while still being powerful enough to handle millions of images on high-end rigs.

### 💡 What is it useful for?
*   **Reference Browsers & Mood Boards:** Instantly find visual references using natural language or rough sketches.
*   **Asset Management:** Organize and search through massive local libraries of images, PDFs, and video frames.

Note: I created this as a tool to help artists quickly search for drawing references through large libraries. 
At some point, looking through tens of thousands of images or videos to locate the exact reference you might remember was there, becomes too time-consuming and an actual chore.
I felt the need to automate all of that and I wanted something very specific, and scalable to millions of records.

My idea is to help with the search process, offering a much smaller subset of candidate references than can be searched by image (even by importing a quick hand-drawn sketch through a webcam) or by natural language, or by booru-style tags. 

Images can also be imported with their own companion metadata files with additional textual descriptions of the contents. 

Although this server can actually be integrated as part of a RAG agent in AI systems, that isn't its intended purpose, and there are better RAGs than this one for agentic use anyways. 


---

## ✨ Core Capabilities
*   **Multi-format Ingestion:** PNG, JPG, TIFF, WebP, PDFs (auto-converted to text + images), CSVs, text/code files, compressed image archives (zip/tar/7z), and Videos (MP4, MKV, AVI).
*   **Smart Metadata:** Automatically associates `.txt` or `.json` files that share a basename with an image.
*   **Versatile Search:** 8 distinct search modes, including text-to-image, image-to-image, sketch-to-image, and video frame search.
*   **On-Demand Delivery:** Returns matching images (or reconstructed PDF/video frames) directly as Base64 in the JSON reply.
*   **Live Monitoring:** Drop files into a watched folder and the agent automatically ingests them in the background.
*   **Front-end Ready:** ZeroMQ REP socket for easy integration with web dashboards, Blender add-ons, or ComfyUI custom nodes.

---

## 🔍 Supported Search Types

| Search Type | Query Modality | Description |
| :--- | :--- | :--- |
| **vector** *(default)* | Text or Image | Tri-vector Reciprocal Rank Fusion across all modalities. |
| **fts** | Text (keywords) | recommended: Pure Full-Text Search. *(Requires building the FTS index first)*. |
| **hybrid** | Text | Combines semantic and keyword retrieval. |
| **textontextvector** | Text | recommended: Semantic text-to-text similarity. |
| **textonimagevector** | Text | recommended: Describe an image in words and find matching visuals. |
| **imageonimagevector** | Image / Sketch | recommended: Search by reference image or a rough sketch. |
| **textonvideovector** | Text | recommended: Find video frames matching a textual description. |
| **imageonvideovector** | Image | recommended: Find video frames visually similar to a reference image. |

---

## 📦 Requirements

**System-level:**
*   Python 3.10+
*   **FFmpeg** (Required for video ingestion. Must be on your system `PATH`).
*   *Optional but recommended:* CUDA-capable GPU (falls back to CPU automatically).

**Python Packages:**
`PySide6`, `pyzmq`, `lancedb`, `pyarrow`, `numpy`, `pillow`, `torch`, `torchvision`, `sentence-transformers`, `transformers`, `huggingface_hub`, `pymupdf`, `psutil`, `py7zr`.

---

## 🛠️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/paoloinverse/art-ref-RAG-server.git
cd art-ref-RAG-server
```

**2. Create and activate a virtual environment**
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**3. Install dependencies**
```bash
python -m pip install --upgrade pip
pip install PySide6 pyzmq lancedb pyarrow numpy pillow pymupdf psutil py7zr
pip install sentence-transformers transformers huggingface_hub
```

*Tip for NVIDIA GPU users: Install the CUDA build of PyTorch that matches your driver (e.g., CUDA 12.1):*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**4. Install FFmpeg**
*   **Debian/Ubuntu:** `sudo apt install ffmpeg`
*   **macOS:** `brew install ffmpeg`
*   **Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add the `bin` folder to your `PATH`.

---

## 🚀 Running the Application

With your virtual environment activated, run:
```bash
python rag_agent_gui12z2_clip-vit_fast.py
```
The GUI will open. By default, it starts with the **Dummy model** so you can test the ingestion and search pipeline immediately without downloading AI weights.

### 🧠 Available CLIP-ViT Models
You can download and switch between models directly in the GUI. The models are autodownloaded from Huggingface.

| Model | Vector Dim | Best For |
| :--- | :--- | :--- |
| **Dummy** | 512 | Testing the pipeline without downloading weights. |
| **CLIP ViT-B-32** | 512 | Fast, lightweight, battle-tested. Best starting point for English. |
| **CLIP ViT-B-32 multilingual** | 512 | **Recommended** for Japanese or non-English textual searches / metadata. |
| **CLIP ViT-L-14** | 768 | **Recommended** for maximum accuracy. This is now my default choice.*(Note: Requires a new table due to 768 dim).* |

---

## 🔌 Network / Front-end Integration

The server exposes a ZeroMQ REP socket (Default: `127.0.0.1:5001`). Send JSON requests and receive JSON replies.

**Ping**
```json
{ "action": "ping" }
```

**Search**
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
*(Valid `search_type` values: `vector`, `fts`, `hybrid`, `textontextvector`, `textonimagevector`, `imageonimagevector`, `textonvideovector`, `imageonvideovector`)*

**List & Load Tables**
```json
{ "action": "list_tables" }
{ "action": "load_table", "table_name": "illustrations_v2" }
```

---

## 📸 Screenshots
*(GUI screenshots will be added here soon)*
<!--
> ![Main Window](screenshots/main_window.png)
> ![Batch Ingestion](screenshots/batch_ingestion.png)
> ![Video Pipeline](screenshots/video_pipeline.png)
> ![Hybrid Search Results](screenshots/hybrid_search.png)
-->

---

## ⚖️ License
Apache-2.0
