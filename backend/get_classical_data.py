import music21
from pathlib import Path

def download_bach_chorales():
    # Make sure the folder exists
    output_dir = Path("dataset/classical")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Searching MIT's music21 corpus for Bach Chorales...")
    # This grabs the internal file paths for all Bach pieces in the library
    bach_paths = music21.corpus.getComposer('bach')
    
    print(f"✅ Found {len(bach_paths)} pieces. Extracting the first 100 as MIDI...")
    
    success_count = 0
    # We will just grab 100 to keep it fast and perfectly sized for your RTX 3050
    for i, path in enumerate(bach_paths[:100]):
        try:
            # Parse the sheet music and write it directly to a .mid file
            score = music21.corpus.parse(path)
            midi_filename = output_dir / f"bach_chorale_{i:03d}.mid"
            score.write('midi', fp=str(midi_filename))
            success_count += 1
        except Exception as e:
            continue # Skip any weird files

    print(f"🎉 Done! Successfully extracted {success_count} multi-track MIDI files into {output_dir}")

if __name__ == "__main__":
    download_bach_chorales()