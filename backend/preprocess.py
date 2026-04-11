import os
import json
from pathlib import Path
from miditok import REMI, TokenizerConfig
from symusic import Score

def build_tokenizer():
    # 1. We inject our custom GENRE SWITCH tokens into the AI's vocabulary
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=True,
        use_programs=True, 
        use_tempos=True,
        special_tokens=["[CLASSICAL]", "[LOFI]", "[ROCK]"]  # The magic buttons
    )
    tokenizer = REMI(config)
    return tokenizer

def process_seq2seq():
    tokenizer = build_tokenizer()
    # We create a new folder so we don't mix up Phase 1 and Phase 2 data
    token_dir = Path("tokens_seq2seq")
    token_dir.mkdir(exist_ok=True)
    
    genres = ["classical", "lofi", "rock"]
    
    # Bulletproof Genre IDs (These numbers act as our magic buttons)
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
                
                # We need at least 2 tracks to have a "Melody" and a "Band"
                if len(midi.tracks) < 2:
                    continue
                    
                # 2. Find the Melody (The track with the highest average pitch)
                highest_pitch = -1
                melody_idx = 0
                
                for idx, track in enumerate(midi.tracks):
                    if len(track.notes) > 0:
                        avg_pitch = sum(note.pitch for note in track.notes) / len(track.notes)
                        if avg_pitch > highest_pitch:
                            highest_pitch = avg_pitch
                            melody_idx = idx
                            
                # 3. Create two blank sheets of music
                prompt_score = Score(midi.ticks_per_quarter)
                target_score = Score(midi.ticks_per_quarter)
                
                # 4. Put the Melody on one sheet, and the Band on the other
                for idx, track in enumerate(midi.tracks):
                    if idx == melody_idx:
                        prompt_score.tracks.append(track)
                    else:
                        target_score.tracks.append(track)
                        
                # 5. Translate both to math
                prompt_tokens = tokenizer(prompt_score).ids
                target_tokens = tokenizer(target_score).ids
                
                # Flatten the lists if miditok outputs 2D arrays
                if isinstance(prompt_tokens[0], list): 
                    prompt_tokens = [item for sublist in prompt_tokens for item in sublist]
                if isinstance(target_tokens[0], list): 
                    target_tokens = [item for sublist in target_tokens for item in sublist]
                
                # 6. THE ALCHEMY: Inject the Genre Tag at the start of the Band's music
                target_tokens.insert(0, genre_ids[genre])
                
                # 7. Save the Seq2Seq Pair
                out_file = token_dir / f"{genre}_{i:03d}.json"
                with open(out_file, 'w') as f:
                    json.dump({
                        "genre": genre, 
                        "prompt": prompt_tokens, # The Melody Input
                        "target": target_tokens  # The Genre Tag + Band Output
                    }, f)
                    
                success_count += 1
                
            except Exception as e:
                # Skip corrupted or incompatible files quietly
                continue
                
        print(f"✅ Successfully created {success_count} Seq2Seq pairs for {genre}.")
                
    print(f"\n🎉 Phase 2 Tokenization Complete! Check the 'tokens_seq2seq' folder.")

if __name__ == "__main__":
    process_seq2seq()