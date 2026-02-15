
import argparse
import os
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw

# Add project root to path
sys.path.append(os.getcwd())

from src.ai_client import call_gemini_3_flash_vision


def _extract_json_object(raw_text: str) -> dict:
    """Extract and parse the first JSON object from model output."""
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        # Best-effort recovery for common truncation cases.
        # Example: {"ymin": 0.7, "xmin": 0.8, "ymax": 1
        if cleaned.startswith("{"):
            repaired = cleaned
            if "xmax" not in repaired:
                repaired = repaired.rstrip(", \n\t")
                repaired += ', "xmax": 1'
            if not repaired.endswith("}"):
                repaired += "}"
            return json.loads(repaired)
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)
    return json.loads(match.group(0))


def test_gemini_crop_single(image_path: Path, vod_id: str, image_name: str) -> None:
    """Test Gemini crop on a single image."""
    print(f"\n{'='*80}")
    print(f"VOD ID: {vod_id}")
    print(f"Testing with image: {image_path}")
    print(f"{'='*80}\n")

    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
    except Exception as e:
        print(f"Failed to read image: {e}")
        return

    images = [("image", img_bytes, "image/jpeg")]
    
    # Single clear prompt - use JSON mode for structured output
    prompt = (
        "Detect the bounding box of the streamer's webcam/face camera in this gaming stream screenshot. "
        "Return a JSON object with normalized coordinates (0.0 to 1.0): "
        '{"ymin": <top>, "xmin": <left>, "ymax": <bottom>, "xmax": <right>}'
    )
    
    bbox: dict | None = None
    last_response = ""

    for attempt in range(1, 4):
        print(f"Calling Gemini 3 Flash Vision (attempt {attempt})...")
        try:
            response = call_gemini_3_flash_vision(
                prompt,
                images,
                max_tokens=500,  # Increased for complete JSON
                temperature=0.0,
                request_tag=f"test_gemini_face_crop_attempt_{attempt}",
                json_mode=(attempt >= 2),  # Try JSON mode on retry
            )
        except Exception as e:
            print(f"Gemini call failed on attempt {attempt}: {e}")
            continue

        last_response = response or ""
        print(f"Raw response (attempt {attempt}): {last_response}")
        if not response:
            continue

        try:
            bbox = _extract_json_object(response)
            break
        except json.JSONDecodeError:
            continue

    if bbox is None:
        print("Failed to parse JSON response after retries.")
        return

    try:
        print(f"Parsed bbox: {bbox}")

        # Crop and save
        with Image.open(image_path) as img:
            w, h = img.size
            ymin = float(bbox.get("ymin", 0))
            xmin = float(bbox.get("xmin", 0))
            ymax = float(bbox.get("ymax", 1))
            xmax = float(bbox.get("xmax", 1))

            ymin = max(0.0, min(1.0, ymin))
            xmin = max(0.0, min(1.0, xmin))
            ymax = max(0.0, min(1.0, ymax))
            xmax = max(0.0, min(1.0, xmax))

            if ymax <= ymin or xmax <= xmin:
                print("Invalid bbox coordinates returned by model.")
                return

            left = xmin * w
            top = ymin * h
            right = xmax * w
            bottom = ymax * h

            print(f"Pixel coords: left={left}, top={top}, right={right}, bottom={bottom}")

            output_dir = Path("data/temp/gemini_face_crop_test")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Draw box on original
            draw = ImageDraw.Draw(img)
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            detection_path = output_dir / f"{vod_id}_{image_name}_detection.jpg"
            img.save(detection_path)
            print(f"✓ Saved detection visualization to {detection_path}")

            # Save crop
            crop = img.crop((left, top, right, bottom))
            crop_path = output_dir / f"{vod_id}_{image_name}_crop.jpg"
            crop.save(crop_path)
            print(f"✓ Saved crop to {crop_path}")

    except json.JSONDecodeError:
        print("Failed to parse JSON response")
        if last_response:
            print(f"Last raw response: {last_response}")
    except Exception as e:
        print(f"Error processing image: {e}")

def test_multiple_images() -> None:
    """Test Gemini crop on multiple images from different VOD IDs."""
    test_images = [
        ("2667506900", "data/temp/arc_cam_snaps/2667506900/bg_arc_000_31825.jpg"),
        ("2668385852", "data/temp/arc_cam_snaps/2668385852/bg_arc_000_19302.jpg"),
        ("2673875461", "data/temp/arc_cam_snaps/2673875461/bg_arc_001_24153.jpg"),
        ("2674588114", "data/temp/arc_cam_snaps/2674588114/bg_arc_000_4870.jpg"),
        ("2696457555", "data/temp/arc_cam_snaps/2696457555/bg_arc_000_2102.jpg"),
    ]
    
    print(f"\n{'#'*80}")
    print(f"# Testing Gemini Face Crop on {len(test_images)} images from different VODs")
    print(f"{'#'*80}\n")
    
    results_summary = []
    
    for vod_id, image_path_str in test_images:
        image_path = Path(image_path_str)
        if not image_path.exists():
            print(f"⚠ Image not found: {image_path}")
            results_summary.append(f"❌ VOD {vod_id}: Image not found")
            continue
        
        image_name = image_path.stem
        try:
            test_gemini_crop_single(image_path, vod_id, image_name)
            results_summary.append(f"✓ VOD {vod_id}: {image_name}")
        except Exception as e:
            print(f"❌ Error processing {vod_id}/{image_name}: {e}")
            results_summary.append(f"❌ VOD {vod_id}: Error - {e}")
    
    # Print summary
    print(f"\n{'#'*80}")
    print(f"# RESULTS SUMMARY")
    print(f"{'#'*80}\n")
    for result in results_summary:
        print(result)
    print(f"\n{'#'*80}")
    print(f"# All outputs saved to: data/temp/gemini_face_crop_test/")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gemini 3 Flash webcam face crop.")
    parser.add_argument("--image", default="", help="Optional image path to test.")
    parser.add_argument("--multi", action="store_true", help="Test multiple images from different VODs.")
    args = parser.parse_args()
    
    if args.multi:
        test_multiple_images()
    else:
        # Original single image test
        base_dir = Path("data/temp/arc_cam_snaps")
        image_path = Path(args.image) if args.image else None
        if image_path and not image_path.exists():
            print(f"Specified image does not exist: {image_path}")
            exit(1)

        if image_path is None:
            for f in base_dir.rglob("bg_*.jpg"):
                image_path = f
                break

        if not image_path:
            print("No test image found in data/temp/arc_cam_snaps")
            exit(1)
        
        vod_id = image_path.parent.name
        image_name = image_path.stem
        test_gemini_crop_single(image_path, vod_id, image_name)
