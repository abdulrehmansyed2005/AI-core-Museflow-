import music21
from pathlib import Path

def download_bach_chorales():
    # Ensure that the output directory for classical music exists
    output_dir = Path("dataset/classical")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Searching for Bach Chorales")
    # Retrieve file paths for Bach musical pieces from the internal library corpus
    bach_paths = music21.corpus.getComposer('bach')
    
    print(f"Found pieces Extracting the first one hundred as midi files")
    
    success_count = 0
    # Select a limited number of pieces to maintain efficient processing speed
    for i, path in enumerate(bach_paths[:100]):
        try:
            # Parse the musical notation and export it as a midi file
            score = music21.corpus.parse(path)
            midi_filename = output_dir / f"bach_chorale_{i:03d}.mid"
            score.write('midi', fp=str(midi_filename))
            success_count += 1
        except Exception as e:
            # Ignore files that fail to parse correctly
            continue

    print(f"Done Successfully extracted midi files into output directory")

if __name__ == "__main__":
    download_bach_chorales()