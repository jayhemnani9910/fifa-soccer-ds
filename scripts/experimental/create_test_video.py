"""Create a synthetic test video for pipeline testing."""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def create_synthetic_video(
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration_seconds: int = 3,
    num_objects: int = 5,
) -> None:
    """Generate a synthetic video with moving colored rectangles simulating players."""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, (width, height))
    
    total_frames = fps * duration_seconds
    
    # Initialize random objects (simulating players)
    np.random.seed(42)
    objects = []
    for i in range(num_objects):
        obj = {
            'x': np.random.randint(50, width - 100),
            'y': np.random.randint(50, height - 100),
            'vx': np.random.randint(-5, 5),
            'vy': np.random.randint(-5, 5),
            'color': tuple(np.random.randint(0, 255, 3).tolist()),
            'size': np.random.randint(40, 80)
        }
        objects.append(obj)
    
    log.info(f"Generating {total_frames} frames at {fps} FPS...")
    
    for frame_idx in range(total_frames):
        # Create green field background
        frame = np.ones((height, width, 3), dtype=np.uint8) * np.array([34, 139, 34], dtype=np.uint8)
        
        # Draw field lines
        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)
        cv2.circle(frame, (width // 2, height // 2), 100, (255, 255, 255), 2)
        cv2.rectangle(frame, (20, 20), (width - 20, height - 20), (255, 255, 255), 2)
        
        # Update and draw objects
        for obj in objects:
            # Update position
            obj['x'] += obj['vx']
            obj['y'] += obj['vy']
            
            # Bounce off walls
            if obj['x'] < 0 or obj['x'] > width - obj['size']:
                obj['vx'] *= -1
            if obj['y'] < 0 or obj['y'] > height - obj['size']:
                obj['vy'] *= -1
            
            # Keep within bounds
            obj['x'] = max(0, min(width - obj['size'], obj['x']))
            obj['y'] = max(0, min(height - obj['size'], obj['y']))
            
            # Draw object
            cv2.rectangle(
                frame,
                (int(obj['x']), int(obj['y'])),
                (int(obj['x'] + obj['size']), int(obj['y'] + obj['size'])),
                obj['color'],
                -1
            )
            
            # Add border
            cv2.rectangle(
                frame,
                (int(obj['x']), int(obj['y'])),
                (int(obj['x'] + obj['size']), int(obj['y'] + obj['size'])),
                (255, 255, 255),
                2
            )
        
        out.write(frame)
    
    out.release()
    log.info(f"Synthetic video created at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create synthetic test video")
    parser.add_argument("--output", default="data/raw/sample.mp4", help="Output video path")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration", type=int, default=3, help="Duration in seconds")
    parser.add_argument("--objects", type=int, default=5, help="Number of moving objects")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    create_synthetic_video(
        output_path=Path(args.output),
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_seconds=args.duration,
        num_objects=args.objects,
    )


if __name__ == "__main__":
    main()
