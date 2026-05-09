"""
MuseFlow Web API
Flask server that wraps the generation pipeline for the web UI.
"""


import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from pathlib import Path
from werkzeug.utils import secure_filename

# --- Model Architecture (must match train.py) ---
class MuseFlowTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        embedded = self.embedding(x)
        out = self.transformer(embedded, mask=mask)
        return self.fc_out(out)


# --- Sampling ---
def sample_next_token(logits, temperature=0.9, top_k=50, top_p=0.95):
    logits = logits / temperature
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k)
        min_top_k = top_k_values[-1]
        logits[logits < min_top_k] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# --- Melody Looper ---
def loop_melody(raw_tokens, target_length=128):
    real_tokens = [t for t in raw_tokens if t != 0]
    if len(real_tokens) == 0:
        return [0] * target_length
    if len(real_tokens) >= target_length:
        return real_tokens[:target_length]
    looped = []
    while len(looped) < target_length:
        looped.extend(real_tokens)
    return looped[:target_length]


# --- Duration Humanizer ---
def humanize_durations(score, jitter_pct=0.15):
    for track in score.tracks:
        for note in track.notes:
            original_dur = note.duration
            if original_dur > 0:
                jitter = random.uniform(1.0 - jitter_pct, 1.0 + jitter_pct)
                note.duration = max(1, int(original_dur * jitter))
            onset_drift = int(original_dur * random.uniform(-0.05, 0.05))
            note.time = max(0, note.time + onset_drift)
    return score


# --- MIDI Cleaner & Quantizer ---
def clean_melody_score(score):
    """
    Sanitizes raw MIDI (especially from basic_pitch) to match training data:
    1. Pick the track with the most notes.
    2. Filter out 'noise' (very short notes).
    3. Quantize to 8th notes (approx 120 ticks at 480 PPQ).
    """
    if len(score.tracks) == 0:
        return score
        
    # 1. Pick the busiest track
    best_track = max(score.tracks, key=lambda t: len(t.notes))
    score.tracks = [best_track]
    
    # 2. Filter & Quantize
    ticks_per_8th = (score.ticks_per_quarter or 480) // 2
    
    new_notes = []
    for note in best_track.notes:
        if note.duration < 40: continue # Skip noise
        
        # Snap onset and duration to 8th note grid
        note.time = round(note.time / ticks_per_8th) * ticks_per_8th
        note.duration = max(ticks_per_8th, round(note.duration / ticks_per_8th) * ticks_per_8th)
        new_notes.append(note)
    
    best_track.notes = new_notes
    return score


def quantize_score(score, grid_division=4):
    """
    Snaps all notes in the score to a grid.
    grid_division=4 means 16th notes (4 per quarter).
    """
    ticks_per_quarter = score.ticks_per_quarter or 480
    grid_ticks = ticks_per_quarter // grid_division
    
    for track in score.tracks:
        for note in track.notes:
            # Snap start time
            note.time = round(note.time / grid_ticks) * grid_ticks
            # Snap duration (minimum 1 grid unit)
            note.duration = max(grid_ticks, round(note.duration / grid_ticks) * grid_ticks)
            # Flatten velocity slightly to avoid 'stabs'
            note.velocity = int(60 + (note.velocity * 0.4))
            
    return score


def normalize_score(score):
    """
    Fixes instruments playing out of order by:
    1. Removing tracks with no notes.
    2. Finding the global earliest note onset across ALL tracks.
    3. Shifting every track by that offset so all instruments start at tick 0.
    4. Sorting notes within each track chronologically.
    """
    # Drop empty tracks
    score.tracks = [t for t in score.tracks if len(t.notes) > 0]
    if not score.tracks:
        return score

    # Find the earliest onset across all tracks
    global_start = min(
        note.time
        for track in score.tracks
        for note in track.notes
    )

    # Shift all notes so the earliest onset is at tick 0
    if global_start != 0:
        for track in score.tracks:
            for note in track.notes:
                note.time = max(0, note.time - global_start)

    # Sort notes within each track by onset time
    for track in score.tracks:
        track.notes.sort(key=lambda n: n.time)

    return score


def consolidate_tracks(score):
    """
    REMI decoding fragments each program change into a new track, so the
    same instrument ends up as many 1-note tracks.  This merges all tracks
    that share the same MIDI program number into one, keeping all their notes.
    Tracks with fewer than 3 notes after merging are dropped as noise.
    """
    from collections import defaultdict
    program_map = defaultdict(list)   # program -> [Track, ...]

    for track in score.tracks:
        prog = getattr(track, 'program', 0) or 0
        program_map[prog].append(track)

    new_tracks = []
    for prog, tracks in program_map.items():
        # Use the first track as the base and pool all notes into it
        base = tracks[0]
        for extra in tracks[1:]:
            base.notes.extend(extra.notes)
        base.notes.sort(key=lambda n: n.time)
        if len(base.notes) >= 3:          # drop near-empty tracks
            new_tracks.append(base)

    score.tracks = new_tracks
    return score





# --- Flask App ---
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Load Model Once at Startup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MuseFlow API] Device: {device.type.upper()}")

# Calculate vocab size
vocab_size = 0
token_files = sorted(Path("tokens_seq2seq").glob("*.json"))
for file in token_files[:100]:  # Sample first 100 for speed
    with open(file, 'r') as f:
        content = json.load(f)
        vocab_size = max(vocab_size, max(content['prompt'] + content['target']) + 1)

# Scan remaining files in batches
for file in token_files[100:]:
    with open(file, 'r') as f:
        content = json.load(f)
        local_max = max(content['prompt'] + content['target']) + 1
        if local_max > vocab_size:
            vocab_size = local_max

print(f"[MuseFlow API] Vocabulary size: {vocab_size}")

model = MuseFlowTransformer(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load("museflow_translator.pth", map_location=device, weights_only=True))
model.eval()
print(f"[MuseFlow API] Model loaded! ({sum(p.numel() for p in model.parameters()):,} params)")


# --- Routes ---

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "ready",
        "device": device.type,
        "vocab_size": vocab_size,
        "model_params": sum(p.numel() for p in model.parameters())
    })


@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        genre = request.form.get('genre', 'classical').lower()
        num_loops = int(request.form.get('loops', 4))
        num_loops = max(1, min(8, num_loops))
        temperature = float(request.form.get('temperature', 0.8)) # Lowered default for rhythm

        genre_tags = {"classical": 10000, "lofi": 10001, "rock": 10002}
        genre_tag = genre_tags.get(genre, 10000)

        # Check for uploaded MIDI
        melody_tokens = None
        if 'melody' in request.files:
            file = request.files['melody']
            if file.filename:
                filename = secure_filename(file.filename)
                filepath = UPLOAD_DIR / filename
                file.save(str(filepath))

                # Tokenize the uploaded MIDI
                from miditok import REMI, TokenizerConfig
                from symusic import Score

                config = TokenizerConfig(
                    num_velocities=32, use_chords=True,
                    use_programs=True, use_tempos=True
                )
                tokenizer = REMI(config)

                if filename.endswith('.wav') or filename.endswith('.mp3'):
                    # Transcribe audio to MIDI first
                    from basic_pitch.inference import predict_and_save
                    from basic_pitch import ICASSP_2022_MODEL_PATH
                    
                    # Delete old transcription if it exists (basic_pitch won't overwrite)
                    midi_path = UPLOAD_DIR / f"{Path(filename).stem}_basic_pitch.mid"
                    if midi_path.exists():
                        midi_path.unlink()
                    
                    try:
                        predict_and_save(
                            audio_path_list=[str(filepath)],
                            output_directory=str(UPLOAD_DIR),
                            save_midi=True, sonify_midi=False,
                            save_model_outputs=False, save_notes=False,
                            model_or_model_path=ICASSP_2022_MODEL_PATH
                        )
                    except UnicodeEncodeError:
                        pass  # basic_pitch emoji print failed on Windows — MIDI was already saved
                    melody_score = Score(str(midi_path))
                else:
                    melody_score = Score(str(filepath))

                # ✨ CLEAN & QUANTIZE custom melody
                melody_score = clean_melody_score(melody_score)

                melody_tokens = tokenizer(melody_score).ids
                if isinstance(melody_tokens[0], list):
                    melody_tokens = [t for sub in melody_tokens for t in sub]

        # Fallback: random melody from dataset
        if melody_tokens is None:
            random_file = random.choice(token_files)
            with open(random_file, 'r') as f:
                melody_tokens = json.load(f)['prompt']

        # Loop melody to fill prompt window
        melody_prompt = loop_melody(melody_tokens, target_length=128)

        # Build starting sequence
        # 200 tokens/loop: enough for multi-instrument phrases without overloading
        gen_tokens = num_loops * 200
        current_sequence = melody_prompt + [genre_tag]

        # Generate
        for i in range(gen_tokens):
            context = current_sequence[-512:]
            x = torch.tensor([context], dtype=torch.long).to(device)
            with torch.no_grad():
                predictions = model(x)
            logits = predictions[0, -1, :]
            next_token = sample_next_token(logits, temperature=temperature, top_k=50, top_p=0.95)
            current_sequence.append(next_token)

        # 1. Decode the original melody
        melody_only_tokens = [t for t in melody_prompt if t not in [10000, 10001, 10002] and t != 0]
        melody_score = tokenizer.decode(melody_only_tokens)
        
        # Calculate melody duration (the "Master Window")
        melody_duration = 0
        if melody_score.tracks:
            melody_score.tracks[0].name = "Main Melody"
            if melody_score.tracks[0].notes:
                melody_duration = max(n.time + n.duration for n in melody_score.tracks[0].notes)

        # 2. Decode the generated band
        generated_only = current_sequence[129:]
        band_tokens = [t for t in generated_only if t not in [10000, 10001, 10002] and t != 0]
        band_score = tokenizer.decode(band_tokens)

        # 3. Combine and Constrain to Melody Duration
        for track in band_score.tracks:
            if not track.notes: continue
            
            # Sort and Shift to 0
            track.notes.sort(key=lambda n: n.time)
            first_onset = track.notes[0].time
            for n in track.notes:
                n.time = max(0, n.time - first_onset)
            
            # Loop/Clip to fit the Melody Window
            if melody_duration > 0:
                original_notes = [n for n in track.notes if n.time < melody_duration]
                if not original_notes: continue
                
                track_span = max(n.time + n.duration for n in original_notes)
                new_notes = list(original_notes)
                
                # Loop to fill if band is shorter than melody
                current_end = track_span
                import copy
                while current_end < melody_duration and track_span > 0:
                    for n in original_notes:
                        if current_end + n.time < melody_duration:
                            nn = copy.copy(n)
                            nn.time += current_end
                            new_notes.append(nn)
                    current_end += track_span
                
                # Final Clip: Remove any note that starts or ends after the melody
                track.notes = [n for n in new_notes if n.time < melody_duration]
                for n in track.notes:
                    if n.time + n.duration > melody_duration:
                        n.duration = max(1, melody_duration - n.time)
            
            melody_score.tracks.append(track)
        
        generated_score = melody_score

        # 4. Final Polish
        generated_score = quantize_score(generated_score, grid_division=4)
        generated_score = normalize_score(generated_score)
        generated_score = humanize_durations(generated_score, jitter_pct=0.05)

        # Tempo and Save
        from symusic import Tempo as SyTempo
        if not generated_score.tempos:
            generated_score.tempos.append(SyTempo(0, 120.0))

        output_name = f"MuseFlow_Synced_{genre.upper()}_{random.randint(1000,9999)}.mid"
        output_path = OUTPUT_DIR / output_name
        generated_score.dump_midi(str(output_path))

        # Track info
        tracks = []
        for idx, track in enumerate(generated_score.tracks):
            tracks.append({
                "index": idx,
                "program": track.program if hasattr(track, 'program') else -1,
                "notes": len(track.notes)
            })

        return jsonify({
            "success": True,
            "filename": output_name,
            "genre": genre,
            "loops": num_loops,
            "tokens_generated": gen_tokens,
            "tracks": tracks
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download(filename):
    filename = secure_filename(filename)
    filepath = (OUTPUT_DIR / filename).resolve()
    response = send_file(
        str(filepath),
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=filename
    )
    response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


if __name__ == '__main__':
    print("\n[MuseFlow] Starting server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
