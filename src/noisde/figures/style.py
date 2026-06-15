"""Shared rendering style helpers for manuscript figures."""
from __future__ import annotations

from pathlib import Path


def typography_rc(base_size: float, preset: str) -> dict[str, object]:
    if preset == "times":
        return {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "font.size": base_size,
            "mathtext.fontset": "stix",
        }
    if preset == "sans":
        return {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": base_size,
            "mathtext.fontset": "dejavusans",
        }
    raise ValueError(f"unknown font preset: {preset}")


def strip_trailing_whitespace(path: Path) -> None:
    text = path.read_text()
    path.write_text("\n".join(line.rstrip(" \t") for line in text.splitlines()) + "\n")
