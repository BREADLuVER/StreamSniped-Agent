"""
Gemini-based transcription client that replaces faster-whisper.
Uses Google's Gemini Flash model to transcribe audio with sound effects,
returning the exact same format as the faster-whisper client.
"""

import os
import time
import json
import logging
import typing
from pathlib import Path
from typing import Dict, List, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    genai = None

logger = logging.getLogger(__name__)

def _configure_genai():
    """Configure the Gemini API client."""
    if not genai:
        raise ImportError("google-generativeai package is not installed.")
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set.")
    
    genai.configure(api_key=api_key)

def transcribe_audio_file(audio_path: Path) -> Dict:
    """
    Transcribe an audio file using Gemini Flash and return a format compatible with faster-whisper.
    
    Returns:
        {
            "text": "Full transcribed text...",
            "language": "en",  # Defaulting to en as Gemini detects auto
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world [Laughter]"
                },
                ...
            ]
        }
    """
    _configure_genai()
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Uploading {audio_path.name} to Gemini for transcription...")
    
    try:
        # 1. Upload file
        audio_file = genai.upload_file(path=str(audio_path))
        
        # 2. Wait for processing
        while audio_file.state.name == "PROCESSING":
            time.sleep(1)
            audio_file = genai.get_file(audio_file.name)
            
        if audio_file.state.name == "FAILED":
            raise RuntimeError(f"Gemini failed to process audio file: {audio_file.state.name}")

        # 3. Generate Content
        # We use gemini-2.0-flash if available, else fallback to 1.5-flash
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        model = genai.GenerativeModel(model_name)
        
        prompt = """
        You are a high-precision audio transcriber. 
        Transcribe this audio clip into a JSON format.

        CRITICAL REQUIREMENTS:
        1. **Verbatim Transcription**: Transcribe exactly what is said.
        2. **Sound Effects**: Include significant sounds in brackets within the text, e.g., [Explosion], [Laughter], [Music], [Screaming], [Silence].
        3. **NO Speaker Labels**: Do NOT include "Streamer:", "Voice 1:", etc. Just the text and sound effects.
        4. **Timestamps & Granularity**: Provide precise start and end timestamps. Group short sentences or phrases into logical segments of 3-10 seconds to avoid excessive fragmentation, unless there is a significant pause.
        5. **JSON Output**: The output MUST be a valid JSON object with a single key "segments".

        Format structure:
        {
          "segments": [
            { "start": 0.0, "end": 4.5, "text": "Hey, what's up? [Laughter] Not much, just gaming." },
            { "start": 5.0, "end": 9.2, "text": "We should go over there and check it out." }
          ]
        }
        """

        response = model.generate_content(
            [prompt, audio_file],
            generation_config={"response_mime_type": "application/json"},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # 4. Parse Response
        try:
            result_json = json.loads(response.text)
            segments = result_json.get("segments", [])
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Gemini JSON response: {response.text}")
            # Fallback: try to extract JSON from markdown code blocks if present
            try:
                text = response.text
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                result_json = json.loads(text)
                segments = result_json.get("segments", [])
            except Exception:
                # If total failure, return empty or raw text as one segment
                logger.error("Could not recover JSON from Gemini response.")
                return {
                    "text": response.text,
                    "language": "en",
                    "segments": [{
                        "start": 0.0,
                        "end": 0.0, # Unknown duration
                        "text": response.text
                    }]
                }

        # 5. Cleanup (optional, but good practice to delete files from Gemini cloud)
        try:
            genai.delete_file(audio_file.name)
        except Exception:
            pass

        # 6. Format for compatibility
        # Ensure segments have float start/end and string text
        formatted_segments = []
        full_text_parts = []
        
        for s in segments:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", 0.0))
            text = str(s.get("text", "")).strip()
            
            if text:
                formatted_segments.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
                full_text_parts.append(text)
        
        return {
            "text": " ".join(full_text_parts),
            "language": "en", # Gemini doesn't explicitly return language code in this mode, assume en or auto
            "segments": formatted_segments
        }

    except Exception as e:
        logger.error(f"Gemini transcription error: {e}")
        # Return empty structure on failure to prevent pipeline crash, 
        # or re-raise if you want to stop processing. 
        # For now, re-raising seems safer so we don't save empty data.
        raise e
