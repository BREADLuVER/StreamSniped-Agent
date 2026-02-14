#!/usr/bin/env python3
"""
Test script to print the exact prompt sent to Gemini for Arc Detection.
Usage: python -m story_archs.test_arch_prompt <vod_id> [chunk_index]
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from story_archs.gemini_arc_detection import (
        load_segments_for_chunk,
        format_transcript_for_prompt,
        extract_chat_peaks,
        get_game_for_timestamp,
        load_chapters,
        CHUNK_ARC_DETECTION_PROMPT,
        _format_hms
    )
except ImportError:
    # Fallback if running directly from file without module context
    sys.path.append(str(Path(__file__).parent.parent))
    from story_archs.gemini_arc_detection import (
        load_segments_for_chunk,
        format_transcript_for_prompt,
        extract_chat_peaks,
        get_game_for_timestamp,
        load_chapters,
        CHUNK_ARC_DETECTION_PROMPT,
        _format_hms
    )

def main():
    parser = argparse.ArgumentParser(description="Print the exact prompt fed to AI for Arc Detection")
    parser.add_argument("vod_id", help="VOD ID")
    parser.add_argument("chunk_index", type=int, nargs="?", default=0, help="Chunk index (default: 0)")
    args = parser.parse_args()

    vod_id = args.vod_id
    chunk_index = args.chunk_index

    print(f"Generating prompt for VOD {vod_id}, Chunk {chunk_index}...")

    # 1. Load Data
    try:
        segments, chunk_start, chunk_end = load_segments_for_chunk(
            vod_id, chunk_index, overlap_seconds=900
        )
    except Exception as e:
        print(f"Error loading segments: {e}")
        return

    chapters = load_chapters(vod_id)
    
    # 2. Prepare Context
    current_game = get_game_for_timestamp(chapters, chunk_start)
    transcript = format_transcript_for_prompt(segments)
    chat_peaks = extract_chat_peaks(segments)
    
    # Mock previous context for the test
    previous_context = "This is the first chunk of the stream." if chunk_index == 0 else f"Summary of chunk {chunk_index-1}..."

    # 3. Format Prompt
    prompt = CHUNK_ARC_DETECTION_PROMPT.format(
        previous_context=previous_context,
        current_game=current_game,
        transcript=transcript,
        chat_peaks=json.dumps(chat_peaks, indent=2),
    )

    print("\n" + "="*80)
    print(f" PROMPT SENT TO GEMINI (Chunk {chunk_index})")
    print("="*80 + "\n")
    print(prompt)
    print("\n" + "="*80)
    print(" END OF PROMPT")
    print("="*80 + "\n")

    print(f"Metadata:")
    print(f"  Time Range: {_format_hms(chunk_start)} - {_format_hms(chunk_end)}")
    print(f"  Transcript Length: {len(transcript)} chars")
    print(f"  Chat Peaks: {len(chat_peaks)}")

if __name__ == "__main__":
    main()
