"""Safe lifecycle helpers for API-created analysis output directories."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

_OUTPUT_MARKER = ".fifa-analysis-output"
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")


def validate_output_name(value: str) -> str:
    """Validate a user-provided output label as a single path component."""
    if value in {".", ".."} or _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(
            "output_dir must be a single name containing only letters, digits, '.', '_' or '-'"
        )
    return value


def analysis_output_root(root: str | Path | None = None) -> Path:
    """Return the configured root for API-managed output directories."""
    configured = root if root is not None else os.getenv("ANALYSIS_OUTPUT_ROOT", "outputs")
    return Path(configured).expanduser().resolve()


def create_analysis_output_dir(
    output_name: str | None,
    task_id: str,
    *,
    root: str | Path | None = None,
) -> Path:
    """Create a unique, marked directory that may later be safely removed."""
    label = validate_output_name(output_name) if output_name else "youtube_analysis"
    safe_task_id = validate_output_name(task_id)
    output_root = analysis_output_root(root)
    output_root.mkdir(parents=True, exist_ok=True)

    target = (output_root / f"{label}_{safe_task_id}").resolve()
    if target.parent != output_root:
        raise ValueError("Resolved output directory escapes the configured output root")

    target.mkdir(mode=0o750, exist_ok=False)
    (target / _OUTPUT_MARKER).write_text("managed by fifa-soccer-ds\n", encoding="utf-8")
    return target


def remove_analysis_output_dir(
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> None:
    """Remove an API-created output directory after verifying its ownership marker."""
    output_root = analysis_output_root(root)
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError("Refusing to remove a symbolic-link output directory")

    target = candidate.resolve()
    if target == output_root or not target.is_relative_to(output_root):
        raise ValueError("Refusing to remove a directory outside the configured output root")
    if not (target / _OUTPUT_MARKER).is_file():
        raise ValueError("Refusing to remove an unmanaged output directory")

    shutil.rmtree(target)


__all__ = [
    "analysis_output_root",
    "create_analysis_output_dir",
    "remove_analysis_output_dir",
    "validate_output_name",
]
