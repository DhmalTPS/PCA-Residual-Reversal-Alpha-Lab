"""Load a strategy file exactly as a single-file submission (its own module, vendor base class)."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

from lab.paths import ensure_predictor_importable


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_strategy(path: str | Path):
    """Return the Predictor subclass defined in `path`."""
    ensure_predictor_importable()
    path = Path(path).resolve()
    name = f"strategy_{path.stem}_{file_sha(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # needed for pickling instances
    spec.loader.exec_module(mod)
    from predictor import Predictor  # resolved via ensure_predictor_importable

    classes = [
        obj for obj in vars(mod).values()
        if inspect.isclass(obj) and issubclass(obj, Predictor) and obj is not Predictor
        and obj.__module__ == name
    ]
    if len(classes) != 1:
        raise ValueError(f"{path.name}: expected exactly one Predictor subclass, found {len(classes)}")
    return classes[0]


def with_overrides(cls, overrides: dict):
    """Subclass with class-attribute overrides (hyper-parameters are class attributes by convention)."""
    if not overrides:
        return cls
    for k in overrides:
        if not hasattr(cls, k):
            raise AttributeError(f"{cls.__name__} has no attribute {k}")
    return type(cls.__name__, (cls,), dict(overrides))


def parse_overrides(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        k, _, v = item.partition("=")
        try:
            out[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            out[k.strip()] = v.strip()
    return out
