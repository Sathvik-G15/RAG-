"""Disk and GPU memory management utilities for Kaggle / resource-constrained environments."""

from __future__ import annotations

import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disk space helpers
# ---------------------------------------------------------------------------

def check_disk(path: str | Path = ".", min_gb: float = 2.0) -> dict[str, Any]:
    """Return free disk space (GB) and raise/warn if below thresholds.

    Returns dict with ``free_gb``, ``total_gb``, ``used_gb``, and ``warning`` (if any).
    """
    p = Path(path).resolve()
    while not p.exists() and p.parent != p:
        p = p.parent

    target_path = str(p) if p.exists() else "."
    usage = shutil.disk_usage(target_path)
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    used_gb = usage.used / (1024**3)

    result: dict[str, Any] = {
        "free_gb": round(free_gb, 2),
        "total_gb": round(total_gb, 2),
        "used_gb": round(used_gb, 2),
        "warning": None,
    }

    if free_gb < min_gb:
        msg = f"🛑 CRITICAL: Only {free_gb:.1f} GB free disk (minimum {min_gb:.1f} GB). Risk of ENOSPC!"
        logger.critical(msg)
        result["warning"] = msg
    elif free_gb < 5.0:
        msg = f"⚠️  Low disk: {free_gb:.1f} GB free. Consider cleaning caches."
        logger.warning(msg)
        result["warning"] = msg
    else:
        logger.info("💾 Disk OK: %.1f GB free / %.1f GB total", free_gb, total_gb)

    return result


def safe_save(data: str | bytes, path: str | Path, min_free_gb: float = 1.0) -> None:
    """Write data to *path* only if sufficient disk space is available.

    Raises ``OSError`` proactively instead of letting the kernel hit ENOSPC.
    """
    path = Path(path)
    info = check_disk(str(path.parent), min_gb=0)
    free_gb = info["free_gb"]

    # Estimate write size
    size_gb = len(data.encode("utf-8") if isinstance(data, str) else data) / (1024**3)

    if free_gb - size_gb < min_free_gb:
        raise OSError(
            f"Refusing to write {path.name} ({size_gb * 1024:.1f} MB): "
            f"only {free_gb:.1f} GB free, need {min_free_gb:.1f} GB headroom."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if isinstance(data, str) else "wb"
    with open(path, mode, encoding="utf-8" if mode == "w" else None) as f:
        f.write(data)
    logger.debug("Saved %s (%.1f KB)", path.name, len(data) / 1024 if isinstance(data, (str, bytes)) else 0)


# ---------------------------------------------------------------------------
# HuggingFace cache cleanup
# ---------------------------------------------------------------------------

def cleanup_hf_cache(
    hf_home: str | None = None,
    keep_models: list[str] | None = None,
    dry_run: bool = False,
) -> float:
    """Remove unused HuggingFace model snapshots to reclaim disk space.

    Parameters
    ----------
    hf_home : str, optional
        Path to HF_HOME directory.  Defaults to ``$HF_HOME`` or ``~/.cache/huggingface``.
    keep_models : list[str], optional
        Model repo IDs to keep (e.g. ``["meta-llama/Llama-3.1-8B-Instruct"]``).
        All other downloaded models in the hub cache are deleted.
    dry_run : bool
        If True, only log what would be deleted without actually removing files.

    Returns
    -------
    float
        Gigabytes freed (0.0 on dry run or nothing to clean).
    """
    hf_home = hf_home or os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = Path(hf_home) / "hub"
    if not hub_dir.exists():
        logger.info("No HF hub cache found at %s", hub_dir)
        return 0.0

    keep_dirs: set[str] = set()
    if keep_models:
        for model_id in keep_models:
            # HF stores as "models--org--name"
            dir_name = f"models--{model_id.replace('/', '--')}"
            keep_dirs.add(dir_name)

    freed_bytes = 0
    for item in hub_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name in keep_dirs:
            logger.info("  Keeping: %s", item.name)
            continue
        if item.name.startswith("models--"):
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            freed_gb = size / (1024**3)
            if dry_run:
                logger.info("  [DRY RUN] Would delete: %s (%.2f GB)", item.name, freed_gb)
            else:
                shutil.rmtree(item, ignore_errors=True)
                logger.info("  Deleted: %s (%.2f GB)", item.name, freed_gb)
            freed_bytes += size

    freed_gb = freed_bytes / (1024**3)
    logger.info("HF cache cleanup: %.2f GB %s", freed_gb, "would be freed" if dry_run else "freed")
    return freed_gb


# ---------------------------------------------------------------------------
# GPU / Python memory cleanup
# ---------------------------------------------------------------------------

def cleanup_gpu_memory(log_prefix: str = "") -> None:
    """Run garbage collection and clear CUDA cache (if available)."""
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated() / (1024**3)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            after = torch.cuda.memory_allocated() / (1024**3)
            if log_prefix:
                logger.info(
                    "%s GPU cleanup: %.2f GB → %.2f GB allocated",
                    log_prefix, before, after,
                )
    except ImportError:
        pass


def cleanup_after_step(step_name: str, path: str = ".") -> None:
    """Combined cleanup: GC + CUDA cache + disk status log.

    Call this between heavy pipeline steps (e.g. after model loading, after each benchmark).
    """
    cleanup_gpu_memory(log_prefix=f"[{step_name}]")
    info = check_disk(path, min_gb=1.0)
    logger.info(
        "[%s] Post-cleanup disk: %.1f GB free / %.1f GB total",
        step_name, info["free_gb"], info["total_gb"],
    )


# ---------------------------------------------------------------------------
# Progress / ETA helper
# ---------------------------------------------------------------------------

def format_progress(current: int, total: int, elapsed_s: float) -> str:
    """Return a progress string like ``[12/50] 24.0% — ETA: 38 min``."""
    pct = (current / total * 100) if total else 0.0
    if current > 0 and elapsed_s > 0:
        avg_per_item = elapsed_s / current
        remaining = avg_per_item * (total - current)
        if remaining > 3600:
            eta = f"{remaining / 3600:.1f} hr"
        elif remaining > 60:
            eta = f"{remaining / 60:.0f} min"
        else:
            eta = f"{remaining:.0f} sec"
    else:
        eta = "calculating..."
    return f"[{current}/{total}] {pct:.1f}% — ETA: {eta}"
