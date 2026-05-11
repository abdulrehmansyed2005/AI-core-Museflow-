# MuseFlow: Deep Technical Architecture & Data Pipeline Guide

This is the comprehensive technical reference for the MuseFlow AI Band Arrangement Engine. It details every step from raw audio signals to the final multi-track MIDI production.

---

## 📂 1. System Architecture Overview

### Backend Framework
*   **Engine:** Python 3.10+
*   **Deep Learning:** PyTorch with CUDA acceleration.
*   **Web Layer:** Flask with Flask-CORS for cross-origin frontend communication.
*   **MIDI Engine:** `symusic` (C++ backend) and `miditok` (REMI encoding).

### Hardware Requirements
*   **Training:** NVIDIA GPU (RTX 3050+ recommended) with `torch.amp` for mixed-precision math.
*   **Inference:** Works on CPU, but optimized for GPU execution (latency < 10s).

---

## 🌊 2. The Detailed Data Journey

### Step 1: Input Acquisition & Transcription
When a user records a melody in the browser:
1.  **Audio Upload:** The `.wav` or `.mp3` is sent to `/api/generate`.
2.  **Neural Transcription:** `app.py` invokes `basic_pitch.inference`. This uses a CNN-based model to detect onsets, offsets, and pitches, outputting a raw MIDI file.
3.  **Cleanup (`clean_melody_score`):**
    *   **Track Selection:** If the transcription creates multiple tracks, we select the one with the most notes.
    *   **Noise Filtering:** Notes shorter than 40 ticks (noise) are deleted.
    *   **Quantization:** All notes are snapped to an 8th-note grid (240 ticks at 480 PPQ) to ensure the AI recognizes the rhythm clearly.

### Step 2: The Prompt Window (`loop_melody`)
The Transformer requires a fixed context length of **128 tokens** for the prompt.
- If the user's input is too short (e.g., a 2-second hum), the system **loops** the sequence until it reaches exactly 128 tokens.
- This creates a "motif" effect, ensuring the AI sees a repeating pattern to harmonize against.

### Step 3: Genre Conditioning (The 10k Tokens)
To steer the AI style, we inject a "Special Token" at the end of the melody:
- `10000`: Classical
- `10001`: Lofi
- `10002`: Rock
The model is trained to associate these tokens with specific instrument distributions (e.g., `10002` triggers Electric Guitars).

### Step 4: Transformer Inference
The `MuseFlowTransformer` is a **4-layer, 8-head Causal Transformer**.
- **Context Size:** 512 tokens (128 prompt + 384 generated).
- **Sampling Strategy:** 
    *   **Temperature (0.8):** Controls randomness (higher = more experimental).
    *   **Top-K (50):** Only allows the 50 most likely next tokens.
    *   **Top-P (0.95):** "Nucleus Sampling" – only keeps the top tokens whose cumulative probability exceeds 95%.
- **Autoregression:** The model predicts one token, adds it to the sequence, and uses that new sequence to predict the next.

### Step 5: Post-Processing Pipeline
Raw tokens are decoded back into MIDI, but require "Studio Polish":
1.  **Syncing:** The generated band is clipped or looped to match the exact duration of the user's input melody.
2.  **Consolidation (`consolidate_tracks`):** The AI often outputs instruments track-by-track. This function merges all tracks sharing the same MIDI Program (e.g., merging 5 guitar fragments into 1 guitar track).
3.  **Normalization (`normalize_score`):** Shifts all tracks so they begin exactly at Tick 0.
4.  **Humanization (`humanize_durations`):** 
    *   Applies ±15% duration jitter.
    *   Applies ±5% onset drift.
    *   This prevents the "robotic" feel of perfectly quantized music.

---

## 🛠️ 3. Deep File Breakdown

### `backend/app.py` (The Central Controller)
*   `MuseFlowTransformer`: Class defining the neural network.
*   `sample_next_token`: Implementation of the Top-K/Top-P math.
*   `/api/generate`: The main logic pipeline that ties transcription, inference, and polish together.

### `backend/train.py` (The Optimizer)
*   **Loss Function:** CrossEntropyLoss with `ignore_index=0` (ignores padding).
*   **Optimizer:** AdamW with Weight Decay (0.01) to prevent the model from getting stuck in "musical loops."
*   **Scheduler:** CosineAnnealingLR (gradually lowers the learning rate as training nears completion).

### `backend/preprocess.py` (The Translator)
*   Uses `REMI` tokenization.
*   **REMI Structure:** `Position` -> `Program` -> `Pitch` -> `Velocity` -> `Duration`.
*   This structured "language" allows the AI to understand that a "Piano" (Program 0) is playing a "C4" (Pitch 60).

### `backend/rescue.py` (The Lofi Logic)
*   Specifically designed for Lofi files that often come as a single track.
*   Uses **Pitch Thresholds** to split one track into three:
    *   **Melody:** >72 (High register)
    *   **Pad:** 48–71 (Mid register)
    *   **Bass:** <48 (Low register)

---

## 🎹 4. Understanding REMI Tokens
Team members should know that the AI doesn't see "notes." It sees a sequence of numbers:
- **0:** Padding (Silence)
- **1-16:** Time positions in a bar.
- **17-144:** Note pitches (Piano keys).
- **10000+:** Genre tags.

---

## 📊 5. Evaluation Metrics
We use a **Confusion Matrix** (`backend/confusion_matrix.png`) to measure how well the model distinguishes genres.
- **Accuracy:** The model is currently ~85% accurate at maintaining genre consistency over a 30-second generation window.
- **Diversity:** Measured by the number of unique instruments (Programs) present in the `outputs/` folder.
