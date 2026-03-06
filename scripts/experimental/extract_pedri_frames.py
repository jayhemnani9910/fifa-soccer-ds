"""Extract frames from sample2.mp4 using different backend."""

import cv2
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def extract_frames_alternative(video_path, output_dir, max_frames=50):
    """Try alternative frame extraction method."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try with different backends
    backends = [
        (cv2.CAP_FFMPEG, "FFmpeg"),
        (cv2.CAP_ANY, "Any"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
    ]
    
    cap = None
    for backend, name in backends:
        log.info(f"Trying {name} backend...")
        cap = cv2.VideoCapture(str(video_path), backend)
        if cap.isOpened():
            log.info(f"✅ {name} backend opened successfully!")
            break
        else:
            log.warning(f"❌ {name} backend failed")
            cap = None
    
    if not cap or not cap.isOpened():
        raise RuntimeError("Could not open video with any backend")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    log.info(f"Video: {width}x{height}, {fps:.2f} FPS, {frame_count} frames")
    
    # Extract frames
    saved = 0
    frame_idx = 0
    stride = max(1, frame_count // max_frames)
    
    log.info(f"Extracting every {stride} frames (max {max_frames} frames)...")
    
    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % stride == 0:
            output_path = output_dir / f"frame_{saved:06d}.jpg"
            # Resize to standard resolution
            resized = cv2.resize(frame, (1280, 720))
            cv2.imwrite(str(output_path), resized)
            saved += 1
            if saved % 10 == 0:
                log.info(f"Extracted {saved} frames...")
        
        frame_idx += 1
    
    cap.release()
    log.info(f"✅ Saved {saved} frames to {output_dir}")
    return saved

if __name__ == "__main__":
    try:
        count = extract_frames_alternative(
            "data/raw/sample2.mp4",
            "data/processed/sample2",
            max_frames=50
        )
        print(f"\n🎉 SUCCESS: Extracted {count} frames!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
