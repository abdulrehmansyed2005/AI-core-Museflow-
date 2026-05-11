import torch
import torch.nn as nn
import torch.nn.functional as F
from miditok import REMI, TokenizerConfig
import json
import sys
import random
from pathlib import Path

# The brain architecture defines the neural network model structure
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

# Sampling helper function for choosing next musical tokens with probability filtering
def sample_next_token(logits, temperature=0.9, top_k=50, top_p=0.95):
    """
    Replaces greedy argmax with temperature plus top k plus top p nucleus sampling
    This lets the model pick Program Change and other rare but important tokens
    """
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
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token


# Melody looper repeats short melodies to fill the input prompt window
def loop_melody(raw_tokens, target_length=128):
    """
    Takes a short melody and LOOPS it
    to fill the full prompt window instead of padding with silence
    
    This way the AI sees a repeating motif and builds a full length arrangement
    around the loop not a one shot phrase followed by dead air
    """
    # Strip out any padding zeros — we only want real musical tokens
    real_tokens = [t for t in raw_tokens if t != 0]
    
    if len(real_tokens) == 0:
        # Nothing to loop — return padded zeros
        return [0] * target_length
    
    if len(real_tokens) >= target_length:
        # Melody is already long enough, just trim
        return real_tokens[:target_length]
    
    # LOOP: tile the melody until we fill the window
    looped = []
    while len(looped) < target_length:
        looped.extend(real_tokens)
    
    looped = looped[:target_length]
    
    num_loops = target_length // len(real_tokens)
    print(f"  Melody looped {num_loops} times")
    
    return looped


# Duration humanizer adds subtle timing variations to make the band sound natural
def humanize_durations(score, jitter_pct=0.15):
    """
    Post processes the decoded MIDI so the band rhythm does not clone
    the melody exact durations Adds jitter random variation to
    every note duration and a slight timing drift to onsets
    
    This makes the output feel like a real band playing together
    not a quantized copy of the input melody
    """
    for track in score.tracks:
        for note in track.notes:
            # Jitter the duration value
            original_dur = note.duration
            if original_dur > 0:
                jitter = random.uniform(1.0 - jitter_pct, 1.0 + jitter_pct)
                note.duration = max(1, int(original_dur * jitter))
            
            # Slight onset drift clamped to avoid negative times
            onset_drift = int(original_dur * random.uniform(-0.05, 0.05))
            note.time = max(0, note.time + onset_drift)
    
    return score


def generate_band():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the musical tokenizer with standard settings
    config = TokenizerConfig(num_velocities=32, use_chords=True, use_programs=True, use_tempos=True)
    tokenizer = REMI(config)

    # Calculate the total vocabulary size from existing data files
    vocab_size = 0
    token_files = list(Path("tokens_seq2seq").glob("*.json"))
    for file in token_files:
        with open(file, 'r') as f:
            content = json.load(f)
            vocab_size = max(vocab_size, max(content['prompt'] + content['target']) + 1)

    # Load the trained translation model weights into memory
    model = MuseFlowTransformer(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load("museflow_translator.pth", weights_only=True))
    model.eval()

        # Load melody input from a file or the dataset directory
    if len(sys.argv) > 1:
        midi_path = sys.argv[1]
        print(f"\n🎤 Loading custom melody from: {midi_path}")
        from symusic import Score
        melody_score = Score(midi_path)
        melody_tokens = tokenizer(melody_score).ids
        if isinstance(melody_tokens[0], list):
            melody_tokens = [item for sublist in melody_tokens for item in sublist]
    else:
        with open(token_files[0], 'r') as f:
            melody_tokens = json.load(f)['prompt']
        print("\nMelody loaded from token files")

        # Repeat the melody to create a continuous input loop for the model
    melody_prompt = loop_melody(melody_tokens, target_length=128)

        # Ask for the desired music genre style
    print("Which genre should the AI Band play?")
    choice = input("Type 'lofi', 'classical', or 'rock': ").strip().lower()
    
    genre_tags = {"classical": 10000, "lofi": 10001, "rock": 10002}
    genre_tag = genre_tags.get(choice, 10000)  # Default to classical if invalid genre is chosen
    if choice not in genre_tags:
        print(f"  Unknown genre chosen defaulting to classical")
        choice = "classical"

        # Determine the length of the generated song in loops
    print("\nHow many loops/bars do you want? (default: 4)")
    try:
        num_loops = int(input("Enter number (1-8): ").strip())
        num_loops = max(1, min(8, num_loops))
    except (ValueError, EOFError):
        num_loops = 4 # Default to four loops if an error occurs during input
    
    # Scale generation tokens: ~150 tokens per loop/section
    gen_tokens = num_loops * 150
    print(f"\nAI Band is generating an arrangement")

        # Combine the looped melody and the genre tag into a single sequence
    current_sequence = melody_prompt + [genre_tag]

        # Main generation loop creates new musical tokens one by one
    for i in range(gen_tokens):
        context = current_sequence[-512:]
        x = torch.tensor([context], dtype=torch.long).to(device)
        
        with torch.no_grad():
            predictions = model(x)
        
        logits = predictions[0, -1, :]
        next_token = sample_next_token(logits, temperature=0.9, top_k=50, top_p=0.95)
        current_sequence.append(next_token)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{gen_tokens} tokens...")

        # Convert the generated tokens back into a midi file and apply humanization
    print("🎵 Translating math back to MIDI...")
    
    final_audio_tokens = [t for t in current_sequence if t not in [10000, 10001, 10002]]
    
    try:
        generated_score = tokenizer.decode(final_audio_tokens)
        
        # 🥁 Humanize durations so the band doesn't clone the melody's rhythm
        generated_score = humanize_durations(generated_score, jitter_pct=0.15)
        print("  Duration humanization applied with jitter")
        
        output_name = f"AI_Band_{choice.upper()}_Output.mid"
        generated_score.dump_midi(output_name)
        print(f"\n✅ SUCCESS! Check your folder for '{output_name}'")
        print(f"   Tracks in output: {len(generated_score.tracks)}")
        for idx, track in enumerate(generated_score.tracks):
            prog_name = f"Program {track.program}" if hasattr(track, 'program') else "Unknown"
            print(f"   Track info displayed here")
    except Exception as e:
        print(f"⚠️ Decoding error: {e}")

if __name__ == "__main__":
    generate_band()