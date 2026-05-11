# MuseFlow: Team Architecture & Workflow Guide

This document explains the technical structure of the MuseFlow AI Band Arrangement Engine and traces the journey of a musical note from recording to final production.

---

## 📂 1. Directory Structure

| Path | Purpose |
| :--- | :--- |
| `backend/` | The core AI engine, model weights, and server logic. |
| `frontend/` | The web interface (HTML/CSS/JS) for user interaction. |
| `dataset/` | Raw MIDI files categorized into Classical, Lofi, and Rock folders. |
| `tokens_seq2seq/` | Processed data pairs used for training the model. |
| `uploads/` | Temporary storage for user-recorded audio or MIDI uploads. |
| `outputs/` | The finalized band arrangements ready for download. |

---

## 🌊 2. The Global Workflow (Input to Output)

### Phase 1: User Input
The process starts in the **Frontend** (`frontend/index.html`).
- The user uploads a MIDI file or records a melody.
- The request is sent to the **Flask API** (`backend/app.py`) via the `/api/generate` route.

### Phase 2: Signal Processing & Cleaning
If the input is **Audio** (.wav/.mp3), `basic_pitch` transcribes it into MIDI.
- **Cleaning:** The raw MIDI is passed through `clean_melody_score()`. It picks the most active track and snaps everything to an 8th-note grid to remove timing errors.
- **Looping:** Since a hummed melody is usually short, `loop_melody()` tiles the input to fill a 128-token window, giving the AI a "repeating motif" to build upon.

### Phase 3: AI Inference (The Transformer)
- **Tokenization:** The melody is converted into REMI tokens (integers).
- **Steering:** A **Genre Tag** (10000 for Classical, 10001 for Lofi, 10002 for Rock) is added to the sequence.
- **Generation:** The `MuseFlowTransformer` uses **Nucleus Sampling (Top-P)** and **Top-K** to predict the next 200–800 tokens. It "hallucinates" the band's instruments based on the style tag.

### Phase 4: Post-Production & Sync
The raw tokens are decoded back into a MIDI score.
- **Alignment:** All band tracks are shifted to start at the exact same time as the melody.
- **Constraint:** Any generated notes extending past the melody are clipped or looped to fit the "Master Window."
- **Humanization:** `humanize_durations()` adds small, random timing shifts (±5%) so the band sounds like real people, not a robot.

---

## 🛠️ 3. Key File Breakdown

### Backend Modules
*   **`app.py`**: The "Boss" file. It runs the server, handles the logic flow, and performs all the post-processing steps.
*   **`train.py`**: The "Teacher." This is used to train the model. It handles the Seq2Seq dataset loading and Mixed Precision training.
*   **`preprocess.py`**: The "Factory." It splits raw dataset MIDI into Melody/Band pairs and saves them as JSON tokens.
*   **`download_dataset.py`**: The "Harvester." It downloads thousands of MIDI files and uses an instrument-scoring system to classify them into genres.
*   **`rescue.py`**: The "Mechanic." It takes single-track Lofi files and splits them into separate Melody, Bass, and Pad tracks based on pitch registers.

### Model Files
*   **`museflow_translator.pth`**: The trained brain. Contains the weights that allow the model to "translate" a melody into a band.

---

## 🎹 4. Musical REMI Tokens (The Language)
We use the **REMI** (Revamped MIDI) tokenization scheme. Instead of simple notes, the AI sees:
1.  **Bar/Position:** Tells the AI where we are in the measure.
2.  **Program:** Tells the AI which instrument is playing.
3.  **Pitch/Velocity/Duration:** Tells the AI how to play the note.

---

## 🚀 5. How to Run for the Team

1.  **Start the Backend:** 
    ```powershell
    cd backend
    python app.py
    ```
2.  **Access the UI:**
    Open `http://localhost:5000` in any browser.
3.  **Generate:**
    Upload a melody, pick a genre, and hit "Generate." The finalized MIDI will appear in the `backend/outputs/` folder.
