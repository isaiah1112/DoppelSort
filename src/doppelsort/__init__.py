from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("doppelsort")
except PackageNotFoundError:  # pragma: no cover - fallback used before installation
    __version__ = "0.0.0"


def main(*args, **kwargs):
    from .cli import main as cli_main

    return cli_main(*args, **kwargs)


__all__ = ["main", "__version__"]
