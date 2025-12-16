"""Process Pedri video using imageio (supports more codecs than OpenCV)."""

import logging
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def extract_frames_imageio(video_path, output_dir, max_frames=30, stride=10):
    """Extract frames using imageio which has better codec support."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"📹 Reading video: {video_path}")
    log.info(f"🎯 Target: {max_frames} frames with stride {stride}")
    
    # Read video metadata
    props = iio.improps(video_path)
    log.info(f"Video properties: {props}")
    
    saved = 0
    frame_idx = 0
    
    log.info("🎬 Extracting frames...")
    
    # Read frames
    for frame in iio.imiter(video_path):
        if frame_idx % stride == 0 and saved < max_frames:
            # Convert RGB to BGR for consistency with OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize to standard resolution
            resized = cv2.resize(frame_bgr, (1280, 720))
            
            # Save frame
            output_path = output_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(output_path), resized)
            saved += 1
            
            if saved % 5 == 0:
                log.info(f"  ✅ Extracted {saved}/{max_frames} frames")
        
        frame_idx += 1
        
        if saved >= max_frames:
            break
    
    log.info(f"🎉 Complete! Saved {saved} frames to {output_dir}")
    return saved


if __name__ == "__main__":
    try:
        count = extract_frames_imageio(
            video_path="data/raw/sample2.mp4",
            output_dir="data/processed/pedri",
            max_frames=30,
            stride=30  # Extract every 30th frame
        )
        print(f"\n✅ SUCCESS: Extracted {count} frames from Pedri video!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
