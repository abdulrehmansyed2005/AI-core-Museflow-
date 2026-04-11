from pathlib import Path
from symusic import Score, Track

# General MIDI Program Numbers (these tell DAWs which instrument to use)
LOFI_MELODY_PROGRAM = 4    # Electric Piano 1 — classic lofi keys
LOFI_CHORD_PROGRAM  = 34   # Fingered Electric Bass — lofi bassline
LOFI_PAD_PROGRAM    = 89   # Warm Pad — ambient background

def rescue_lofi():
    print("🚑 Booting up the Lofi Rescuer...")
    folder = Path("dataset/lofi")
    midi_files = list(folder.glob("*.mid")) + list(folder.glob("*.midi"))
    
    success = 0
    for file_path in midi_files:
        try:
            score = Score(file_path)
            
            # If it already has multiple tracks, skip it
            if len(score.tracks) >= 2:
                continue 
                
            # Grab the squished single track
            single_track = score.tracks[0]
            
            # Create tracks WITH instrument assignments
            melody_track = Track(program=LOFI_MELODY_PROGRAM, name="Melody - E.Piano")
            chord_track = Track(program=LOFI_CHORD_PROGRAM, name="Bass")
            pad_track = Track(program=LOFI_PAD_PROGRAM, name="Pad")
            
            # Split the notes by pitch into 3 ranges
            for note in single_track.notes:
                if note.pitch >= 72:        # High register → Melody
                    melody_track.notes.append(note)
                elif note.pitch >= 48:      # Mid register → Pad/Chords
                    pad_track.notes.append(note)
                else:                       # Low register → Bass
                    chord_track.notes.append(note)
            
            # Build the multi-instrument score
            tracks = [melody_track]
            if len(chord_track.notes) > 0:
                tracks.append(chord_track)
            if len(pad_track.notes) > 0:
                tracks.append(pad_track)
            
            # Only save if we actually got at least 2 tracks
            if len(tracks) >= 2:
                score.tracks = tracks
                score.dump_midi(file_path)
                success += 1
                print(f"  ✓ {file_path.name}: {len(tracks)} tracks created")
            
        except Exception as e:
            continue
            
    print(f"🎸 Successfully split {success} Lofi files into multi-instrument tracks!")

if __name__ == "__main__":
    rescue_lofi()