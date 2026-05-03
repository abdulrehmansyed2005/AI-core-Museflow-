# 🎵 MuseFlow

**AI-Powered Multi-Genre Band Arrangement Engine**

MuseFlow is a Transformer-based system that listens to your melody and arranges a full multi-instrument band around it — in **Classical**, **Lofi**, or **Rock** style.

---

## ✨ Features

- 🎤 **Hum-to-MIDI** — Record a melody, transcribe it via Basic Pitch, and feed it to the AI
- 🧠 **Seq2Seq Transformer** — 4-layer, 8-head causal Transformer trained on 6,000 multi-instrument MIDI files
- 🎛️ **Genre Conditioning** — Special genre tokens steer the model toward Classical, Lofi, or Rock arrangements
- 🥁 **Duration Humanizer** — Post-processing adds natural timing jitter so the output doesn't sound robotic
- 🔁 **Melody Looping** — Short inputs are automatically tiled to fill the full context window

---

## 📁 Project Structure

```
MuseFlow/
├── backend/                    # AI engine (Python)
│   ├── download_dataset.py     # Downloads & classifies Lakh MIDI dataset
│   ├── rescue.py               # Splits single-track Lofi into multi-instrument
│   ├── get_classical_data.py   # Downloads Bach chorales from music21
│   ├── preprocess.py           # Tokenizes MIDI → Seq2Seq JSON pairs
│   ├── train.py                # Trains the Transformer model
│   ├── generate.py             # Generates band arrangements from melodies
│   ├── transcribe.py           # Audio → MIDI via Basic Pitch
│   ├── test_gpu.py             # Quick CUDA availability check
│   └── requirements.txt        # Python dependencies
│
├── frontend/                   # Web UI (coming soon)
│   └── README.md
│
├── Museflow-guide.html         # Interactive project guide
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup

```bash
cd backend
python -m venv env
env\Scripts\activate          # Windows
# source env/bin/activate     # macOS / Linux

pip install -r requirements.txt
# For CUDA GPU support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Build the Dataset

```bash
python download_dataset.py    # Downloads Lakh MIDI (~223 MB) & classifies into genres
python rescue.py              # (Optional) Fix single-track Lofi files
```

### 3. Preprocess & Train

```bash
python preprocess.py          # Tokenize into Seq2Seq pairs
python train.py               # Train the Transformer (~25 epochs)
```

### 4. Generate Music

```bash
# From a hummed recording:
python transcribe.py          # Converts hum.wav → hum_basic_pitch.mid
python generate.py hum_basic_pitch.mid

# From any MIDI file:
python generate.py my_melody.mid
```

You'll be prompted to choose a genre (`classical` / `lofi` / `rock`) and loop count (1–8).

---

## 🏗️ Model Architecture

| Parameter | Value |
|-----------|-------|
| Embedding Dim | 256 |
| Attention Heads | 8 |
| Layers | 4 |
| Prompt Length | 128 tokens (melody) |
| Target Length | 384 tokens (band) |
| Optimizer | AdamW (lr=3e-4) |
| LR Schedule | Cosine Annealing |
| Precision | Mixed FP16 |
| Batch Size | 4 (fits 6 GB VRAM) |

---

## 🎯 Genres

| Genre | Genre Tag | Key Instruments |
|-------|-----------|----------------|
| 🎻 Classical | `10000` | Strings, Woodwinds, Brass |
| 🎹 Lofi | `10001` | Electric Piano, Pads, Vibraphone |
| 🎸 Rock | `10002` | Electric Guitar, Distortion, Drums |

---

## 🛠️ Tech Stack

- **PyTorch** — Model training & inference
- **MidiTok (REMI)** — MIDI tokenization
- **SyMusic** — MIDI file I/O & manipulation
- **Basic Pitch** — Audio-to-MIDI transcription (Spotify)
- **music21** — Classical dataset from MIT corpus

---

## 📄 License

This project is for educational and research purposes.
