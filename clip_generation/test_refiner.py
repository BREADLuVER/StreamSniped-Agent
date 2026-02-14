"""
Test script for Gemini Refiner.
Usage: python -m clip_generation.test_refiner <vod_id> [timestamp_seconds]
"""

import sys
import argparse
from pathlib import Path
from .loader import load_docs
from .gemini_refiner import refine_clip_boundaries, _format_transcript_window

def hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    parser = argparse.ArgumentParser(description="Test Gemini Refiner on a VOD")
    parser.add_argument("vod_id", help="VOD ID to test")
    parser.add_argument("timestamp", type=float, nargs="?", help="Optional anchor timestamp (seconds). If not provided, uses a high-energy moment.")
    args = parser.parse_args()

    print(f"Loading docs for {args.vod_id}...")
    try:
        docs = load_docs(args.vod_id)
    except Exception as e:
        print(f"Error loading docs: {e}")
        return

    if not docs:
        print("No docs found.")
        return

    anchor_time = args.timestamp
    
    # If no timestamp provided, find a high chat_rate_z moment
    if anchor_time is None:
        print("Finding a high-energy moment...")
        best_doc = max(docs, key=lambda d: d.chat_rate_z)
        anchor_time = best_doc.start
        print(f"Selected anchor at {anchor_time:.1f}s ({hms(anchor_time)}) (chat_z: {best_doc.chat_rate_z:.2f})")

    # Define a heuristic window (e.g., ±30s) to simulate the pipeline input
    initial_start = max(0.0, anchor_time - 30.0)
    initial_end = anchor_time + 30.0
    
    print(f"\n--- INPUT ---")
    print(f"Anchor: {anchor_time:.1f}s ({hms(anchor_time)})")
    print(f"Heuristic Window: {initial_start:.1f}s - {initial_end:.1f}s ({hms(initial_start)} - {hms(initial_end)})")
    
    # Run Refiner
    print(f"\n--- RUNNING GEMINI REFINER ---")
    
    # Calculate context window (same logic as in gemini_refiner.py)
    context_pad = 160.0
    context_start = max(0.0, initial_start - context_pad)
    context_end = initial_end + context_pad
    
    print(f"Sending ~5 min context to Gemini ({hms(context_start)} - {hms(context_end)})...")
    
    print("\n--- FULL CONTEXT TRANSCRIPT ---")
    print(_format_transcript_window(docs, context_start, context_end))
    print("-------------------------------\n")
    
    new_start, new_end, meta = refine_clip_boundaries(docs, initial_start, initial_end, anchor_time)
    
    print(f"\n--- OUTPUT ---")
    if meta.get("error"):
        print(f"Error: {meta['error']}")
    elif meta.get("status") == "rejected_by_llm":
        print(f"Rejected: {meta.get('reason')}")
    else:
        print(f"Refined Window: {new_start:.1f}s - {new_end:.1f}s ({hms(new_start)} - {hms(new_end)})")
        print(f"Duration: {new_end - new_start:.1f}s")
        print(f"Reasoning: {meta.get('reasoning')}")
        
        print("\n--- REFINED TRANSCRIPT ---")
        print(_format_transcript_window(docs, new_start, new_end))

if __name__ == "__main__":
    main()
