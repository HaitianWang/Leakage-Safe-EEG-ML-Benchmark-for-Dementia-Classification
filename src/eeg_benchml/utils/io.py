"""Filesystem and serialisation helpers.

Helpers in this module are deliberately conservative: they avoid pickling
non-numerical objects and prefer plain JSON / NumPy for portability across
operating systems.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create the directory at ``path`` if it does not exist.

    Parameters
    ----------
    path : str or pathlib.Path
        Target directory path.

    Returns
    -------
    resolved : pathlib.Path
        The resolved directory path.
    """
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def dump_json(payload: Mapping[str, Any], path: PathLike) -> None:
    """Serialise a JSON-compatible mapping with ``indent=2`` and UTF-8."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: PathLike) -> Dict[str, Any]:
    """Load a JSON document into a plain dictionary."""
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _json_default(value: Any) -> Any:
    """Cast NumPy scalars / arrays to native Python types for JSON output."""
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serialisable")
