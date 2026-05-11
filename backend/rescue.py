from pathlib import Path
from symusic import Score, Track

# General midi program identification numbers for lofi instrument assignments
LOFI_MELODY_PROGRAM = 4    # Electric Piano one for classic lofi keys
LOFI_CHORD_PROGRAM  = 34   # Fingered Electric Bass for lofi bassline
LOFI_PAD_PROGRAM    = 89   # Warm Pad for ambient background music

def rescue_lofi():
    print("🚑 Booting up the Lofi Rescuer...")
    folder = Path("dataset/lofi")
    midi_files = list(folder.glob("*.mid")) + list(folder.glob("*.midi"))
    
    success = 0
    for file_path in midi_files:
        try:
            score = Score(file_path)
            
            # Skip files that already contain multiple musical tracks
            if len(score.tracks) >= 2:
                continue 
                
            # Extract the notes from the single track file
            single_track = score.tracks[0]
            
            # Initialize new tracks with specific instrument programs
            melody_track = Track(program=LOFI_MELODY_PROGRAM, name="Melody - E.Piano")
            chord_track = Track(program=LOFI_CHORD_PROGRAM, name="Bass")
            pad_track = Track(program=LOFI_PAD_PROGRAM, name="Pad")
            
            # Divide the notes into three separate tracks based on their pitch register
            for note in single_track.notes:
                if note.pitch >= 72:        # High register for melody
                    melody_track.notes.append(note)
                elif note.pitch >= 48:      # Mid register for pad and chords
                    pad_track.notes.append(note)
                else:                       # Low register for bass lines
                    chord_track.notes.append(note)
            
            # Assemble the new multi instrument score from the divided tracks
            tracks = [melody_track]
            if len(chord_track.notes) > 0:
                tracks.append(chord_track)
            if len(pad_track.notes) > 0:
                tracks.append(pad_track)
            
            # Save the file only if the splitting process was successful
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