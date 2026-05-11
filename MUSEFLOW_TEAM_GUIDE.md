# MuseFlow: Deep Technical Architecture & File Encyclopedia

This document serves as the absolute technical reference for the MuseFlow project. It details the system architecture, data pipeline, and provides an exhaustive breakdown of every file in the repository.

---

## 🌊 1. Global Workflow (The Lifecycle of a Note)

1.  **Input:** User records audio or MIDI in `frontend/index.html`.
2.  **Transcription:** `basic_pitch` in `backend/app.py` converts audio to MIDI.
3.  **Cleaning:** `app.py` quantizes the melody to an 8th-note grid.
4.  **Tokenization:** `miditok` converts MIDI into REMI integers.
5.  **Inference:** `MuseFlowTransformer` predicts a band accompaniment based on a Genre Tag (10000-10002).
6.  **Decoding:** Tokens are converted back to MIDI tracks.
7.  **Polish:** `app.py` merges tracks, syncs durations, and adds ±15% timing jitter.
8.  **Output:** A multi-track MIDI file is served for download.

---

## 📂 2. File-by-File Encyclopedia

### 🏢 Root Directory (Project Management)
*   **`README.md`**: The primary documentation. Contains installation steps, project overview, and quick-start commands.
*   **`project_report.html`**: The formal technical report. Includes architectural diagrams, training statistics (12,000 files), and performance analysis.
*   **`Museflow-guide.html`**: A user-facing manual explaining how to use the web interface effectively.
*   **`MUSEFLOW_TEAM_GUIDE.md`**: (This file) The deep technical reference for team members and developers.
*   **`MuseFlow_Project_Poster.png`**: A high-resolution marketing/academic poster summarizing the project for evaluations.
*   **`.gitignore`**: Tells Git to ignore heavy folders like `dataset/`, `tokens_seq2seq/`, `env/`, and `__pycache__` to keep the repository clean.
*   **`MuseFlow.code-workspace`**: VS Code configuration file that sets up the workspace environment for the team.

### 🎨 `frontend/` (The User Interface)
*   **`index.html`**: The main structure. Includes the recording interface, genre selectors, and the MIDI player visualization.
*   **`style.css`**: Defines the "Glassmorphism" aesthetic. Uses modern CSS variables for dark-mode themes, gradients, and responsive layouts.
*   **`script.js`**: The "brain" of the UI. It handles the `Web Audio API` for recording, sends multi-part form data to the Flask API, and manages the progress bars.

### ⚙️ `backend/` (The AI Engine)

#### **Core Logic & API**
*   **`app.py`**: The most critical file.
    *   **Purpose:** Serves the Flask API and manages the generation pipeline.
    *   **Key Functions:** `clean_melody_score` (Quantizer), `loop_melody` (Tiler), `consolidate_tracks` (Track Merger), `humanize_durations` (Timing Jitter).
*   **`requirements.txt`**: Lists all Python dependencies (`torch`, `flask`, `symusic`, `miditok`, `basic-pitch`).
*   **`museflow_translator.pth`**: The trained model weights. It contains the millions of learned parameters from the 50-epoch training run.

#### **Training & Data Pipeline**
*   **`train.py`**: The training script.
    *   **Logic:** Implements a Seq2Seq training loop with `AdamW` optimization and `CosineAnnealing` learning rate scheduling.
    *   **Features:** Uses `torch.amp` (Mixed Precision) to allow training on consumer GPUs like the RTX 3050.
*   **`preprocess.py`**: The data formatter.
    *   **Logic:** Reads MIDI files, identifies the melody track, and pairs it with the rest of the band. It saves these as JSON token pairs in `tokens_seq2seq/`.
*   **`download_dataset.py`**: The data harvester.
    *   **Logic:** Automatically downloads the Lakh MIDI Dataset and classifies files into genres using a "point-based" instrument scoring system.
*   **`get_classical_data.py`**: A specialized script that uses the `music21` library to extract 100+ Bach chorales to ensure high-quality classical training data.
*   **`rescue.py`**: The Lofi fixer. It takes single-track MIDI files and use pitch-thresholding to "rescue" them into multi-track formats (Melody, Pad, Bass).

#### **Utilities & Testing**
*   **`generate.py`**: A standalone command-line version of the generator. Useful for testing the model without starting the full web server.
*   **`test_gpu.py`**: A simple script to verify that PyTorch can see your NVIDIA GPU and CUDA is properly configured.
*   **`transcribe.py`**: A standalone script to test the `basic_pitch` audio-to-MIDI transcription logic.
*   **`confusion_matrix.png`**: A visual chart showing how accurately the model classifies and generates different genres.

---

## 🧠 3. Deep Neural Logic

### Transformer Parameters
- **Layers:** 4 Encoder layers.
- **Heads:** 8 Attention heads per layer.
- **Embedding Dim:** 256.
- **Sequence Length:** 512 tokens.

### Sampling Math
- **Temperature (0.8):** We use 0.8 because 1.0 was too chaotic and 0.5 was too repetitive. 0.8 provides the best balance of "musical creativity."
- **Top-P (0.95):** This ensures the model never picks a token with near-zero probability, eliminating "sour notes" from the output.

---

## 🎹 4. REMI Token Breakdown
Every note the AI "hears" is translated into these specific tokens:
1.  **Bar/Position:** Grid timing (where in the measure).
2.  **Program:** Which instrument (Piano, Guitar, etc).
3.  **Pitch:** The MIDI note number (0-127).
4.  **Duration:** How long the note is held.
5.  **Genre:** The style "anchor" (10000+).
