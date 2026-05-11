import os
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
import json
from pathlib import Path
from miditok import REMI, TokenizerConfig
from symusic import Score

def build_tokenizer():
    # Inject custom genre switch tokens into the model vocabulary
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=True,
        use_programs=True, 
        use_tempos=True,
        special_tokens=["[CLASSICAL]", "[LOFI]", "[ROCK]"]  # Genre control tokens used for steering the model generation
    )
    tokenizer = REMI(config)
    return tokenizer

def process_seq2seq():
    tokenizer = build_tokenizer()
    # Create a new folder to store tokenized sequence to sequence data
    token_dir = Path("tokens_seq2seq")
    token_dir.mkdir(exist_ok=True)
    
    genres = ["classical", "lofi", "rock"]
    
    # Define unique identification numbers for each music genre
    genre_ids = {
        "classical": 10000, 
        "lofi": 10001,
        "rock": 10002
    }
    
    for genre in genres:
        print(f"\n⚙️ Splitting {genre.upper()} into Melody & Band...")
        folder = Path(f"dataset/{genre}")
        
        if not folder.exists():
            continue
            
        midi_files = list(folder.glob("*.mid")) + list(folder.glob("*.midi"))
        success_count = 0
        
        for i, file_path in enumerate(midi_files):
            try:
                midi = Score(file_path)
                
                # Require at least two tracks for melody and accompaniment splitting
                if len(midi.tracks) < 2:
                    continue
                    
                # Identify the melody as the track with the highest average pitch
                highest_pitch = -1
                melody_idx = 0
                
                for idx, track in enumerate(midi.tracks):
                    if len(track.notes) > 0:
                        avg_pitch = sum(note.pitch for note in track.notes) / len(track.notes)
                        if avg_pitch > highest_pitch:
                            highest_pitch = avg_pitch
                            melody_idx = idx
                            
                # Create blank scores for the melody and band tracks
                prompt_score = Score(midi.ticks_per_quarter)
                target_score = Score(midi.ticks_per_quarter)
                
                # Separate the melody track from the rest of the accompaniment tracks
                for idx, track in enumerate(midi.tracks):
                    if idx == melody_idx:
                        prompt_score.tracks.append(track)
                    else:
                        target_score.tracks.append(track)
                        
                # Convert musical scores into tokenized integer sequences
                prompt_tokens = tokenizer(prompt_score).ids
                target_tokens = tokenizer(target_score).ids
                
                # Flatten the token lists if they are in a multi dimensional format
                if isinstance(prompt_tokens[0], list): 
                    prompt_tokens = [item for sublist in prompt_tokens for item in sublist]
                if isinstance(target_tokens[0], list): 
                    target_tokens = [item for sublist in target_tokens for item in sublist]
                
                # Insert the specific genre identification tag at the beginning of the band sequence
                target_tokens.insert(0, genre_ids[genre])
                
                # Save the processed melody and band pairs as a json file
                out_file = token_dir / f"{genre}_{i:03d}.json"
                with open(out_file, 'w') as f:
                    json.dump({
                        "genre": genre, 
                        "prompt": prompt_tokens, # Melody input sequence for the model
                        "target": target_tokens  # Genre tag and band output sequence from the model
                    }, f)
                    
                success_count += 1
                
            except Exception as e:
                # Ignore files that cannot be processed properly
                continue
                
        print(f"✅ Successfully created {success_count} Seq2Seq pairs for {genre}.")
                
    print(f"\n🎉 Phase 2 Tokenization Complete! Check the 'tokens_seq2seq' folder.")

if __name__ == "__main__":
    process_seq2seq()