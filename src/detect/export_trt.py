"""Prepare TensorRT engines from ONNX exports with safety checks.

This module converts ONNX models to optimized TensorRT engines for low-latency
inference on NVIDIA GPUs. Supports FP32, FP16, and INT8 quantization levels.

Key functions:
    - export_trt: Convert ONNX to TensorRT engine (.plan file)
    - build_engine: Internal builder using TensorRT API
"""

from __future__ import annotations

import gc
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

log = logging.getLogger(__name__)

# Memory management for TensorRT operations
_trt_lock = threading.Lock()

@contextmanager
def _trt_memory_manager():
    """Context manager for TensorRT memory operations with proper cleanup."""
    with _trt_lock:
        try:
            # Clear CUDA cache before operation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            yield
            
        finally:
            # Cleanup after operation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection again
            gc.collect()

class TensorRTMemoryManager:
    """Manages TensorRT object lifecycle and GPU memory cleanup."""
    
    def __init__(self):
        self._objects: Dict[str, Any] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def register_object(self, name: str, obj: Any) -> None:
        """Register a TensorRT object for cleanup."""
        self._objects[name] = obj
        self._logger.debug(f"Registered TensorRT object: {name}")
    
    def cleanup_object(self, name: str) -> None:
        """Clean up a specific TensorRT object."""
        if name in self._objects:
            obj = self._objects.pop(name)
            try:
                # Call cleanup if available
                if hasattr(obj, '__del__'):
                    obj.__del__()
                del obj
                self._logger.debug(f"Cleaned up TensorRT object: {name}")
            except Exception as e:
                self._logger.warning(f"Error cleaning up {name}: {e}")
    
    def cleanup_all(self) -> None:
        """Clean up all registered TensorRT objects."""
        for name in list(self._objects.keys()):
            self.cleanup_object(name)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()


@dataclass(slots=True)
class TensorRTExportConfig:
    """Parameters for converting an ONNX graph into a TensorRT engine."""

    onnx_path: str = "build/yolov8n.onnx"
    output: str = "build/yolov8n_fp16.plan"
    fp16: bool = True
    int8: bool = False
    workspace_size: int = 1 << 30  # 1 GiB


def _require_modules() -> tuple:
    """Validate that TensorRT bindings are available before conversion.

    Returns:
        Tuple of (tensorrt, polygraphy) modules
    """
    try:
        import tensorrt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("TensorRT Python bindings are required for engine export.") from exc

    try:
        import polygraphy.backend.trt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("polygraphy is required for managing TensorRT builds.") from exc

    return tensorrt, polygraphy


def build_engine(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    int8: bool = False,
    workspace_size: int = 1 << 30,
) -> Path:
    """Build a TensorRT engine from ONNX using the native TensorRT API.

    Args:
        onnx_path: Path to input ONNX model
        output_path: Path where .plan engine file will be written
        fp16: Enable FP16 half-precision mode
        int8: Enable INT8 quantization (requires calibration)
        workspace_size: GPU memory workspace in bytes (default: 1 GiB)

    Returns:
        Path to generated .plan file

    Raises:
        ImportError: If TensorRT is not available
        FileNotFoundError: If ONNX file doesn't exist
        RuntimeError: If engine build fails
    """
    import tensorrt

    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_file}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log.info("Building TensorRT engine from %s", onnx_file)
    log.info("Config: fp16=%s, int8=%s, workspace=%d MB", fp16, int8, workspace_size >> 20)

    # Use memory manager for proper cleanup
    with _trt_memory_manager(), TensorRTMemoryManager() as mem_mgr:
        try:
            # Create builder, network, and parser with proper tracking
            logger = tensorrt.Logger(tensorrt.Logger.WARNING)
            mem_mgr.register_object('logger', logger)
            
            builder = tensorrt.Builder(logger)
            mem_mgr.register_object('builder', builder)
            
            network = builder.create_network(
                1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            mem_mgr.register_object('network', network)
            
            parser = tensorrt.OnnxParser(network, logger)
            mem_mgr.register_object('parser', parser)

            # Parse ONNX model
            with onnx_file.open("rb") as f:
                if not parser.parse(f.read()):
                    errors = "\n".join(parser.get_error(i).desc() for i in range(parser.num_errors))
                    raise RuntimeError(f"Failed to parse ONNX: {errors}")

            # Configure builder with memory tracking
            config = builder.create_builder_config()
            mem_mgr.register_object('config', config)
            config.set_memory_pool_limit(tensorrt.MemoryPoolType.WORKSPACE, workspace_size)

            if fp16:
                config.set_flag(tensorrt.BuilderFlag.FP16)
                log.info("Enabled FP16 precision")

            if int8:
                config.set_flag(tensorrt.BuilderFlag.INT8)
                log.info("Enabled INT8 quantization (requires calibration)")

            # Build engine
            log.info("Building engine... this may take a minute")
            engine = builder.build_serialized_network(network, config)
            
            # Clean up intermediate objects before saving
            mem_mgr.cleanup_object('config')
            mem_mgr.cleanup_object('network')
            mem_mgr.cleanup_object('parser')
            mem_mgr.cleanup_object('builder')
            mem_mgr.cleanup_object('logger')
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")

            # Save engine
            with output_file.open("wb") as f:
                f.write(engine)
            
            # Clean up engine object
            del engine
            
            log.info("Engine written to %s (%d MB)", output_file, output_file.stat().st_size >> 20)
            return output_file
            
        except Exception as e:
            log.error(f"TensorRT engine build failed: {e}")
            # Ensure cleanup happens even on error
            raise


def export_trt(config: TensorRTExportConfig) -> Path:
    """Export a TensorRT engine from an ONNX model.

    This function validates the environment and generates a .plan file suitable
    for deployment with TensorRT runtime. Includes proper memory management
    to prevent GPU memory leaks during batch exports.
    """
    _require_modules()

    onnx_file = Path(config.onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_file}")

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap the entire export process in memory management
    with _trt_memory_manager():
        try:
            result = build_engine(
                onnx_path=config.onnx_path,
                output_path=config.output,
                fp16=config.fp16,
                int8=config.int8,
                workspace_size=config.workspace_size,
            )
            
            # Final cleanup after successful build
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()
            
            return result
            
        except Exception as e:
            log.error(f"TensorRT export failed: {e}")
            # Ensure cleanup happens on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()
            raise


def export_batch_trt(configs: list[TensorRTExportConfig]) -> list[Path]:
    """Export multiple TensorRT engines with memory management between exports.
    
    Args:
        configs: List of TensorRTExportConfig objects
        
    Returns:
        List of paths to generated .plan files
        
    Raises:
        RuntimeError: If any export fails
    """
    results = []
    failed_exports = []
    
    log.info(f"Starting batch export of {len(configs)} TensorRT engines")
    
    for i, config in enumerate(configs):
        try:
            log.info(f"Processing export {i+1}/{len(configs)}: {config.onnx_path}")
            
            # Export with memory management
            result = export_trt(config)
            results.append(result)
            
            # Force cleanup between exports to prevent memory accumulation
            with _trt_memory_manager():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
            
            log.info(f"✓ Export {i+1} completed: {result}")
            
        except Exception as e:
            log.error(f"✗ Export {i+1} failed for {config.onnx_path}: {e}")
            failed_exports.append((config.onnx_path, str(e)))
            
            # Continue with next exports but track failures
            continue
    
    if failed_exports:
        error_msg = f"Failed to export {len(failed_exports)} engines:\n"
        for path, error in failed_exports:
            error_msg += f"  - {path}: {error}\n"
        raise RuntimeError(error_msg)
    
    log.info(f"✓ Batch export completed successfully: {len(results)} engines")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser(description="Export ONNX model to TensorRT engine.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--output", required=True, help="Output .plan file path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--workspace", type=int, default=1 << 30, help="Workspace size in bytes")
    parser.add_argument("--batch", action="store_true", help="Enable batch export mode")
    parser.add_argument("--batch-config", help="JSON file with batch export configurations")

    args = parser.parse_args()

    if args.batch and args.batch_config:
        # Batch export mode
        import json
        with open(args.batch_config, 'r') as f:
            batch_configs = json.load(f)
        
        configs = [TensorRTExportConfig(**cfg) for cfg in batch_configs]
        
        try:
            results = export_batch_trt(configs)
            log.info("✓ Batch TensorRT export completed successfully")
            for result in results:
                log.info(f"  Generated: {result}")
        except Exception as exc:  # pragma: no cover
            log.exception("✗ Batch TensorRT export failed: %s", exc)
            raise
    else:
        # Single export mode
        cfg = TensorRTExportConfig(
            onnx_path=args.onnx,
            output=args.output,
            fp16=args.fp16,
            int8=args.int8,
            workspace_size=args.workspace,
        )

        try:
            export_trt(cfg)
            log.info("✓ TensorRT export completed successfully")
        except Exception as exc:  # pragma: no cover
            log.exception("✗ TensorRT export failed: %s", exc)
            raise
