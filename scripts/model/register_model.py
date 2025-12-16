#!/usr/bin/env python3
"""Register trained model to model registry with metadata and metrics.

This script creates a comprehensive model registration package including:
- Model binary with versioning
- Training metrics and performance data
- Model metadata for deployment
- Validation results and artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Register model to model registry.")
    parser.add_argument(
        "--model-path", 
        required=True, 
        help="Path to trained model file."
    )
    parser.add_argument(
        "--model-name", 
        required=True, 
        help="Name for model registration."
    )
    parser.add_argument(
        "--version", 
        required=True, 
        help="Version identifier for this model."
    )
    parser.add_argument(
        "--metrics", 
        help="Path to metrics JSON file."
    )
    parser.add_argument(
        "--registry-dir", 
        default="model_registry",
        help="Model registry directory."
    )
    parser.add_argument(
        "--description",
        default="FIFA soccer detection model",
        help="Model description."
    )
    parser.add_argument(
        "--author",
        default="FIFA DS Team",
        help="Model author or team."
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=["detection", "soccer", "tracking"],
        help="Tags for model categorization."
    )
    return parser.parse_args()


def load_metrics(metrics_path: str | None) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    if not metrics_path or not Path(metrics_path).exists():
        return {}
    
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load metrics from {metrics_path}: {e}")
        return {}


def create_model_metadata(
    model_name: str,
    version: str,
    model_path: Path,
    metrics: Dict[str, Any],
    description: str,
    author: str,
    tags: list[str]
) -> Dict[str, Any]:
    """Create comprehensive model metadata."""
    model_file = Path(model_path)
    
    # Get model file info
    model_size = model_file.stat().st_size if model_file.exists() else 0
    
    return {
        "name": model_name,
        "version": version,
        "description": description,
        "author": author,
        "tags": tags,
        "created_at": datetime.utcnow().isoformat(),
        "model_info": {
            "filename": model_file.name,
            "size_bytes": model_size,
            "size_mb": round(model_size / (1024 * 1024), 2),
            "format": model_file.suffix[1:] if model_file.suffix else "unknown"
        },
        "performance_metrics": metrics,
        "framework": "pytorch",
        "task": "object_detection",
        "input_shape": [720, 1280, 3],  # HWC format
        "classes": ["ball", "player", "referee"],
        "requirements": {
            "python": ">=3.8",
            "pytorch": ">=1.9.0",
            "opencv": ">=4.5.0"
        },
        "deployment": {
            "supported_formats": ["onnx", "tensorrt"],
            "inference_type": "real-time",
            "hardware": ["cpu", "gpu"]
        }
    }


def register_model(
    model_path: str,
    model_name: str,
    version: str,
    metrics_path: str | None,
    registry_dir: str,
    description: str,
    author: str,
    tags: list[str]
) -> Dict[str, Any]:
    """Register model to registry directory."""
    
    model_file = Path(model_path)
    registry_path = Path(registry_dir)
    model_registry_path = registry_path / model_name
    version_path = model_registry_path / version
    
    # Load metrics
    metrics = load_metrics(metrics_path)
    
    # Create metadata
    metadata = create_model_metadata(
        model_name=model_name,
        version=version,
        model_path=model_file,
        metrics=metrics,
        description=description,
        author=author,
        tags=tags
    )
    
    # Create registry directories
    registry_path.mkdir(parents=True, exist_ok=True)
    model_registry_path.mkdir(parents=True, exist_ok=True)
    version_path.mkdir(parents=True, exist_ok=True)
    
    # Copy model file
    model_dest = version_path / model_file.name
    if model_file.exists():
        shutil.copy2(model_file, model_dest)
        log.info(f"Copied model to {model_dest}")
    else:
        log.warning(f"Model file not found: {model_file}")
    
    # Save metadata
    metadata_path = version_path / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save metrics separately
    if metrics:
        metrics_path = version_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Create latest symlink
    latest_path = model_registry_path / "latest"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(version, target_is_directory=True)
    
    # Create registration metrics
    registration_metrics = {
        "registration_timestamp": datetime.utcnow().isoformat(),
        "model_name": model_name,
        "version": version,
        "registry_path": str(version_path),
        "model_size_mb": metadata["model_info"]["size_mb"],
        "metrics_count": len(metrics),
        "registration_success": True
    }
    
    log.info(f"Model registered successfully: {model_name}:{version}")
    log.info(f"Registry location: {version_path}")
    
    return registration_metrics


def main() -> None:
    """Main registration function."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    
    try:
        metrics = register_model(
            model_path=args.model_path,
            model_name=args.model_name,
            version=args.version,
            metrics_path=args.metrics,
            registry_dir=args.registry_dir,
            description=args.description,
            author=args.author,
            tags=args.tags
        )
        
        # Save registration metrics
        metrics_file = Path(args.registry_dir) / "registration_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        log.info("Model registration completed successfully")
        
    except Exception as e:
        log.error(f"Model registration failed: {e}")
        raise


if __name__ == "__main__":
    main()