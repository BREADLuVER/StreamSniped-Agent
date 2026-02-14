"""
Test script to compare Gemini Audio Analysis vs Whisper Transcript.
Downloads a specific audio segment from a VOD and sends it to Gemini for detailed analysis.

Usage: python -m clip_generation.test_audio_refiner <vod_id> [start_seconds] [end_seconds]
Default window: 7692.0 - 7752.0 (from user example)
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path
from typing import List

from .loader import load_docs
from .types import WindowDoc

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_whisper_transcript(docs: List[WindowDoc], start: float, end: float) -> str:
    """Extract the existing Whisper transcript for the window."""
    lines = []
    for d in docs:
        if d.end < start:
            continue
        if d.start > end:
            break
        text = (d.text or "").strip()
        if text:
            lines.append(f"[{hms(d.start)}] {text}")
    return "\n".join(lines)

def download_audio_segment(vod_id: str, start: float, end: float, output_path: Path) -> bool:
    """Download audio segment using TwitchDownloaderCLI."""
    twitch_cli = os.getenv("TWITCH_DOWNLOADER_PATH", "TwitchDownloaderCLI")
    
    # Use HH:MM:SS format like create_individual_clips.py
    start_str = hms(start)
    end_str = hms(end)
    
    cmd = [
        twitch_cli,
        "videodownload",
        "--id", vod_id,
        "-b", start_str,
        "-e", end_str,
        "-o", str(output_path),
        "-q", "Audio", 
        "--ffmpeg-path", "./executables/ffmpeg.exe"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Try Audio quality first
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and output_path.exists():
            return True
            
        # Fallback to 360p if Audio fails (sometimes Audio quality isn't listed)
        print("Audio-only download failed, trying 360p...")
        cmd[7] = "360p"
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and output_path.exists():
            return True
            
        print(f"Download failed: {result.stderr}")
        return False
    except Exception as e:
        print(f"Error downloading: {e}")
        return False

def analyze_with_gemini(audio_path: Path) -> str:
    """Upload audio to Gemini and get a detailed transcript."""
    try:
        import google.generativeai as genai
    except ImportError:
        return "Error: google-generativeai not installed."

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not set."

    genai.configure(api_key=api_key)
    
    print("Uploading audio to Gemini...")
    try:
        # Upload the file
        audio_file = genai.upload_file(path=str(audio_path))
        print(f"Uploaded file: {audio_file.name}")
        
        # Wait for processing
        while audio_file.state.name == "PROCESSING":
            print("Waiting for audio processing...")
            time.sleep(2)
            audio_file = genai.get_file(audio_file.name)
            
        if audio_file.state.name == "FAILED":
            return "Error: Audio processing failed in Gemini."
            
    except Exception as e:
        return f"Error uploading file: {e}"

    prompt = """
    Analyze this audio clip from a livestream. Your goal is to create a script-style transcript that clearly distinguishes speakers.

    CRITICAL INSTRUCTIONS:
    1. **Speaker Diarization**: You MUST distinguish between the main "Streamer" and other voices.
       - The "Streamer" usually has the clearest, highest-quality microphone.
       - "Discord/Voice Chat" usually sounds compressed or lower quality.
       - "Game Characters" often sound scripted or have specific effects.
       - If you hear a different voice but can't identify who it is, label them as "Voice 2", "Voice 3", etc. DO NOT label everyone as "Streamer".
    2. **Sound Effects**: Include significant sounds like [Explosion], [Laughter], [Keyboard], [Silence].
    3. **Tone**: Indicate emotion/tone in parentheses, e.g. (Sarcastic), (Panicked).

    Format:
    [MM:SS] [Sound Effect]
    Streamer: (Tone) Text...
    Voice 2: (Tone) Text...
    Game: Text...
    """

    print("Generating analysis...")
    try:
        # Use Gemini 2.0 Flash (or fallback to 1.5 Flash if 2.0 not available in this lib version yet)
        # Note: google-generativeai supports gemini-2.0-flash-exp if the API supports it.
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        response = model.generate_content([prompt, audio_file])
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare Whisper vs Gemini Audio Analysis")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("start", type=float, nargs="?", default=7692.0, help="Start time (seconds)")
    parser.add_argument("end", type=float, nargs="?", default=7752.0, help="End time (seconds)")
    args = parser.parse_args()

    print(f"--- CONFIG ---")
    print(f"VOD: {args.vod_id}")
    print(f"Window: {args.start}s - {args.end}s ({hms(args.start)} - {hms(args.end)})")
    
    # 1. Get Whisper Transcript
    print(f"\n--- 1. EXISTING WHISPER TRANSCRIPT ---")
    try:
        docs = load_docs(args.vod_id)
        whisper_text = get_whisper_transcript(docs, args.start, args.end)
        print(whisper_text if whisper_text else "[No transcript found in DB for this range]")
    except Exception as e:
        print(f"Error loading docs: {e}")

    # 2. Download Audio
    print(f"\n--- 2. DOWNLOADING AUDIO ---")
    temp_dir = Path("temp_audio_test")
    temp_dir.mkdir(exist_ok=True)
    audio_file_mp4 = temp_dir / f"{args.vod_id}_{int(args.start)}_{int(args.end)}.mp4"
    audio_file_mp3 = temp_dir / f"{args.vod_id}_{int(args.start)}_{int(args.end)}.mp3"
    
    # Download MP4 if MP3 doesn't exist
    if not audio_file_mp3.exists():
        if not audio_file_mp4.exists():
            success = download_audio_segment(args.vod_id, args.start, args.end, audio_file_mp4)
            if not success:
                print("Failed to download audio. Exiting.")
                return
        else:
            print(f"Using existing MP4 file: {audio_file_mp4}")
            
        # Convert to MP3
        print("Converting to MP3...")
        ffmpeg_path = "./executables/ffmpeg.exe"
        cmd = [
            ffmpeg_path,
            "-i", str(audio_file_mp4),
            "-vn", # No video
            "-acodec", "libmp3lame",
            "-q:a", "2", # High quality VBR
            "-y", # Overwrite
            str(audio_file_mp3)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Converted to: {audio_file_mp3}")
            # Optional: remove mp4
            # audio_file_mp4.unlink()
        except subprocess.CalledProcessError as e:
            print(f"Error converting to MP3: {e}")
            return
    else:
        print(f"Using existing MP3 file: {audio_file_mp3}")

    # 3. Analyze with Gemini
    print(f"\n--- 3. GEMINI AUDIO ANALYSIS ---")
    gemini_text = analyze_with_gemini(audio_file_mp3)
    print(gemini_text)
    
    # Cleanup
    # if audio_file.exists():
    #    audio_file.unlink()
    #    temp_dir.rmdir()

if __name__ == "__main__":
    main()
