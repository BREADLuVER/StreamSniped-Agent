"""
Gemini-based refinement for clip boundaries.

This module provides "Micro-Top-Down" analysis for clips. 
Unlike the "Macro" arc detection (which looks at 30-minute chunks), 
this looks at small windows (approx 2-3 minutes) surrounding a detected signal spike
to precisely identify the "Setup" and "Punchline" of a viral moment.
"""

import os
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .types import WindowDoc

# -----------------------------------------------------------------------------
# Prompt Template
# -----------------------------------------------------------------------------

CLIP_REFINE_PROMPT = """You are an expert video editor for a viral clips channel.
Your task is to refine the start and end timestamps for a short clip derived from a raw transcript.

CONTEXT:
- We found a high-energy moment (chat spike / loud audio) at timestamp: {anchor_time_str}
- The transcript below covers the surrounding context (approx 5-6 minutes).

TRANSCRIPT:
{transcript}

YOUR GOAL:
Find the perfect "viral clip" boundaries by understanding the narrative context.
1. **UNDERSTAND**: Read the transcript around the anchor time. What is the funny moment, fail, or story?
2. **SETUP (Start)**: Find where the context begins. Don't start mid-sentence. Don't start too early (boring). Start exactly when the relevant topic/action begins.
3. **PUNCHLINE (End)**: Find where the moment resolves or the reaction peaks. Cut right after the energy dies down or the topic shifts.

OUTPUT (JSON only):
{{
  "start_time": <float seconds>,
  "end_time": <float seconds>,
  "reasoning": "<brief explanation of why you chose these boundaries based on the content>"
}}

RULES:
- **Target Duration**: Ideally keep it under 3 minutes (180s) for YouTube Shorts.
- **Soft Limit**: If the content naturally requires more than 3 minutes (e.g., a long story), DO NOT force a cut. Provide the full natural duration. We will handle it as a video instead of a Short.
- **Precision**: Snap to the start of a sentence for start_time. Snap to the end of a sentence/reaction for end_time.
- If the segment is boring or has no clear clip, return "start_time": -1.
"""

# -----------------------------------------------------------------------------
# Gemini Client
# -----------------------------------------------------------------------------

def call_gemini_flash(prompt: str) -> str:
    """Call Gemini 3 Flash (or configured model) with the prompt."""
    try:
        from google import genai
    except ImportError:
        print("x google-genai not installed. Skipping Gemini refinement.")
        return ""

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("x GEMINI_API_KEY not set. Skipping Gemini refinement.")
        return ""

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_CLIP_MODEL", "gemini-2.0-flash")

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.4,
            }
        )
        return response.text
    except Exception as e:
        print(f"x Gemini API error: {e}")
        return ""

# -----------------------------------------------------------------------------
# Logic
# -----------------------------------------------------------------------------

def _format_transcript_window(docs: List[WindowDoc], start_window: float, end_window: float, chunk_size_seconds: int = 10) -> str:
    """Format transcript lines within the window, aggregating text into time blocks."""
    lines = []
    
    current_block_text = []
    current_block_start = -1.0
    
    # Filter docs first
    relevant_docs = [d for d in docs if d.end >= start_window and d.start <= end_window]
    
    if not relevant_docs:
        return ""
        
    current_block_start = relevant_docs[0].start

    for d in relevant_docs:
        text = (d.text or "").strip()
        if not text:
            continue
            
        # If this segment pushes us past the chunk size, flush the buffer
        if (d.end - current_block_start) > chunk_size_seconds and current_block_text:
            # Flush current block
            block_content = " ".join(current_block_text)
            lines.append(f"({current_block_start:.1f}s) {block_content}")
            
            # Reset for next block
            current_block_text = []
            current_block_start = d.start
            
        current_block_text.append(text)
        
    # Flush remaining text
    if current_block_text:
        block_content = " ".join(current_block_text)
        lines.append(f"({current_block_start:.1f}s) {block_content}")
    
    return "\n".join(lines)

def refine_clip_boundaries(
    docs: List[WindowDoc], 
    initial_start: float, 
    initial_end: float, 
    anchor_time: float
) -> Tuple[float, float, Dict]:
    """
    Refine clip boundaries using Gemini 3 Flash.
    
    Args:
        docs: Full list of WindowDocs (transcript segments).
        initial_start: Heuristic start time.
        initial_end: Heuristic end time.
        anchor_time: The peak energy timestamp.
        
    Returns:
        (refined_start, refined_end, metadata_dict)
        If refinement fails, returns (initial_start, initial_end, {}).
    """
    # 1. Define a "Context Window" significantly larger than the heuristic clip
    #    to give the LLM room to expand if we missed the setup.
    #    Target: ~5-6 minutes total window (±150-180s)
    context_pad = 160.0  # Look ~2.5 mins before/after the heuristic window
    window_start = max(0.0, initial_start - context_pad)
    window_end = initial_end + context_pad
    
    transcript_text = _format_transcript_window(docs, window_start, window_end)
    
    if not transcript_text or len(transcript_text) < 50:
        return initial_start, initial_end, {"error": "empty_transcript"}

    # 2. Build Prompt
    anchor_str = f"{int(anchor_time // 60):02d}:{int(anchor_time % 60):02d}"
    prompt = CLIP_REFINE_PROMPT.format(
        anchor_time_str=anchor_str,
        transcript=transcript_text
    )

    # 3. Call LLM
    response_json = call_gemini_flash(prompt)
    if not response_json:
        return initial_start, initial_end, {"error": "llm_call_failed"}

    # 4. Parse
    try:
        data = json.loads(response_json)
        ref_start = float(data.get("start_time", -1))
        ref_end = float(data.get("end_time", -1))
        
        # Validation
        if ref_start < 0 or ref_end < 0:
            # LLM rejected the clip
            return initial_start, initial_end, {"status": "rejected_by_llm", "reason": data.get("reasoning")}
            
        if ref_end <= ref_start:
            return initial_start, initial_end, {"error": "invalid_timestamps"}
            
        # Sanity check: Don't stray TOO far from the anchor (e.g. > 3 mins away)
        if abs(ref_start - anchor_time) > 180 or abs(ref_end - anchor_time) > 180:
             return initial_start, initial_end, {"error": "hallucinated_timestamps"}

        return ref_start, ref_end, {
            "status": "refined",
            "title": data.get("title"),
            "category": data.get("category"),
            "reasoning": data.get("reasoning")
        }

    except Exception as e:
        print(f"x JSON parse error in refine_clip: {e}")
        return initial_start, initial_end, {"error": "json_parse_error"}
