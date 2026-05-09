"""
MuseFlow Dataset Downloader & Genre Classifier
Downloads the Lakh MIDI Clean subset and filters into genre folders.
Only keeps multi-instrument files (≥2 tracks, ≥2 distinct programs).
"""

import os
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
import shutil
import re
import tarfile
import urllib.request
from pathlib import Path

from symusic import Score

# --- CONFIGURATION ---
LAKH_URL = "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"
DOWNLOAD_DIR = Path("_lakh_download")
EXTRACT_DIR = DOWNLOAD_DIR / "clean_midi"
TAR_FILE = DOWNLOAD_DIR / "clean_midi.tar.gz"

DATASET_DIR = Path("dataset")
TARGET_PER_GENRE = 4000

# Minimum requirements — relaxed to catch more files
MIN_TRACKS = 2          # At least 2 tracks with notes
MIN_PROGRAMS = 2        # At least 2 different instruments (no piano-only)
MIN_NOTES = 30          # Don't want near-empty files
MAX_NOTES = 15000       # Don't want absurdly huge files


def download_lakh():
    """Download the Lakh MIDI Clean subset (~223MB)."""
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    
    if TAR_FILE.exists():
        size_mb = TAR_FILE.stat().st_size / (1024 * 1024)
        print(f"✅ Archive already downloaded: {TAR_FILE} ({size_mb:.0f} MB)")
        return
    
    print(f"⬇️  Downloading Lakh MIDI Clean...")
    print(f"   URL: {LAKH_URL}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r   {mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="", flush=True)
    
    urllib.request.urlretrieve(LAKH_URL, TAR_FILE, reporthook=progress_hook)
    print(f"\n✅ Download complete!")


def sanitize_filename(name):
    """Remove characters that Windows can't handle in file paths."""
    name = re.sub(r'[<>:"|?*]', '_', name)
    name = name.encode('ascii', 'replace').decode('ascii').replace('?', '_')
    return name


def extract_lakh():
    """Extract the tar.gz archive, skipping files with bad names."""
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("*.mid")):
        midi_count = len(list(EXTRACT_DIR.rglob("*.mid")))
        print(f"✅ Already extracted: {midi_count} MIDI files found")
        return
    
    print(f"📦 Extracting archive (sanitizing filenames for Windows)...")
    extracted = 0
    skipped = 0
    with tarfile.open(TAR_FILE, "r:gz") as tar:
        for member in tar.getmembers():
            try:
                clean_name = sanitize_filename(member.name)
                member.name = clean_name
                tar.extract(member, DOWNLOAD_DIR)
                extracted += 1
            except Exception:
                skipped += 1
                continue
    
    midi_count = len(list(EXTRACT_DIR.rglob("*.mid")))
    print(f"✅ Extracted {extracted} items ({skipped} skipped), {midi_count} MIDI files found")


def classify_genre(programs, has_drums):
    """
    Classify genre based on instrument programs.
    Piano (prog 0) is allowed in ALL genres — the key is what ELSE is there.
    """
    # Check for genre-defining instruments
    has_electric_guitar = bool(programs & set(range(25, 32)))
    has_distortion = bool(programs & {29, 30})
    has_bass = bool(programs & set(range(32, 40)))
    has_strings = bool(programs & set(range(40, 52)))
    has_brass = bool(programs & set(range(56, 64)))
    has_woodwinds = bool(programs & set(range(64, 80)))
    has_epiano = bool(programs & {4, 5})
    has_pads = bool(programs & set(range(88, 96)))
    has_synth_lead = bool(programs & set(range(80, 88)))
    has_organ = bool(programs & set(range(16, 24)))
    has_acoustic_guitar = bool(programs & {24, 25})
    has_vibes = bool(programs & {11, 12, 14})
    
    rock_score = 0
    classical_score = 0
    lofi_score = 0
    
    # --- ROCK: Electric guitars are the strongest signal ---
    if has_electric_guitar:
        rock_score += 6
    if has_distortion:
        rock_score += 5
    if has_drums and has_electric_guitar:
        rock_score += 3
    if has_bass and has_electric_guitar:
        rock_score += 2
    if has_organ and has_electric_guitar:
        rock_score += 2
    if has_drums and has_bass and not has_strings:
        rock_score += 1
    
    # --- CLASSICAL: Strings + orchestral instruments ---
    if has_strings:
        classical_score += 4
    if has_woodwinds:
        classical_score += 4
    if has_brass and not has_electric_guitar:
        classical_score += 3
    if has_strings and has_woodwinds:
        classical_score += 3
    if has_strings and not has_electric_guitar and not has_drums:
        classical_score += 3
    if has_acoustic_guitar and has_strings:
        classical_score += 1
    
    # --- LOFI: Electric piano, pads, soft instruments ---
    if has_epiano:
        lofi_score += 5
    if has_pads:
        lofi_score += 4
    if has_vibes:
        lofi_score += 3
    if has_epiano and has_bass and not has_electric_guitar:
        lofi_score += 3
    if has_synth_lead or has_pads:
        lofi_score += 2
    if has_acoustic_guitar and not has_electric_guitar:
        lofi_score += 1
    
    scores = {"classical": classical_score, "rock": rock_score, "lofi": lofi_score}
    best_genre = max(scores, key=lambda k: scores[k])
    
    if scores[best_genre] == 0:
        return "unknown"
    
    return best_genre


def filter_and_classify():
    """
    Scan all extracted MIDI files, filter for multi-instrument quality,
    and classify into genres.
    """
    for genre in ["classical", "lofi", "rock"]:
        (DATASET_DIR / genre).mkdir(parents=True, exist_ok=True)
    
    all_midis = list(EXTRACT_DIR.rglob("*.mid")) + list(EXTRACT_DIR.rglob("*.midi"))
    print(f"\n🔍 Scanning {len(all_midis)} MIDI files...")
    
    genre_counts = {"classical": 0, "lofi": 0, "rock": 0}
    skipped = {"too_few_tracks": 0, "too_few_programs": 0, "too_few_notes": 0, 
               "too_many_notes": 0, "unknown_genre": 0, "genre_full": 0, "error": 0}
    
    for i, midi_path in enumerate(all_midis):
        if (i + 1) % 2000 == 0:
            print(f"   Scanned {i+1}/{len(all_midis)} | "
                  f"C={genre_counts['classical']}, "
                  f"L={genre_counts['lofi']}, "
                  f"R={genre_counts['rock']}")
        
        if all(c >= TARGET_PER_GENRE for c in genre_counts.values()):
            print(f"   🎯 All genres hit {TARGET_PER_GENRE}! Stopping.")
            break
        
        try:
            score = Score(midi_path)
            
            # Filter 1: Track count
            tracks_with_notes = [t for t in score.tracks if len(t.notes) > 0]
            if len(tracks_with_notes) < MIN_TRACKS:
                skipped["too_few_tracks"] += 1
                continue
            
            # Filter 2: Instrument diversity (no piano-only!)
            programs = set()
            total_notes = 0
            has_drums = False
            for track in tracks_with_notes:
                programs.add(track.program)
                total_notes += len(track.notes)
                if track.is_drum:
                    has_drums = True
            
            if len(programs) < MIN_PROGRAMS:
                skipped["too_few_programs"] += 1
                continue
            
            # Filter 3: Note count
            if total_notes < MIN_NOTES:
                skipped["too_few_notes"] += 1
                continue
            if total_notes > MAX_NOTES:
                skipped["too_many_notes"] += 1
                continue
            
            # Classify genre
            genre = classify_genre(programs, has_drums)
            
            if genre == "unknown":
                # Fallback: assign to whichever genre needs files most
                need = {g: TARGET_PER_GENRE - c for g, c in genre_counts.items() if c < TARGET_PER_GENRE}
                if need:
                    genre = max(need, key=lambda k: need[k])
                else:
                    skipped["genre_full"] += 1
                    continue
            
            if genre_counts[genre] >= TARGET_PER_GENRE:
                skipped["genre_full"] += 1
                continue
            
            # Copy to dataset
            dest = DATASET_DIR / genre / f"lakh_{genre}_{genre_counts[genre]:04d}.mid"
            shutil.copy2(midi_path, dest)
            genre_counts[genre] += 1
            
        except Exception:
            skipped["error"] += 1
            continue
    
    # Report
    print(f"\n{'='*50}")
    print(f"📊 DATASET BUILD COMPLETE")
    print(f"{'='*50}")
    for genre, count in genre_counts.items():
        status = "✅" if count >= TARGET_PER_GENRE else "⚠️"
        print(f"  {status} {genre:12s}: {count:5d} / {TARGET_PER_GENRE}")
    print(f"\n  Skipped:")
    for reason, count in skipped.items():
        if count > 0:
            print(f"    {reason:20s}: {count}")
    print(f"{'='*50}")
    
    return genre_counts


def main():
    print("🎵 MuseFlow Dataset Builder")
    print("=" * 50)
    print(f"Target: {TARGET_PER_GENRE} multi-instrument files per genre")
    print(f"Genres: classical, lofi, rock")
    print(f"Filter: ≥{MIN_TRACKS} tracks, ≥{MIN_PROGRAMS} instruments, {MIN_NOTES}-{MAX_NOTES} notes")
    print("=" * 50)
    
    download_lakh()
    extract_lakh()
    counts = filter_and_classify()
    
    total = sum(counts.values())
    print(f"\n🎉 Done! {total} files ready in dataset/ folder.")
    print(f"   Next step: python preprocess.py")


if __name__ == "__main__":
    main()
