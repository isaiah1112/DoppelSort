# Project Guidelines

## Project design

- DoppelSort is a Python 3.12 CLI. The `doppelsort` command is registered in `pyproject.toml`; `src/doppelsort/__main__.py` also supports `uv run python -m doppelsort`.
- `src/doppelsort/cli.py` contains the Click commands and face-indexing, comparison, and SQLite operations. Keep CLI changes consistent across command help, output, and the underlying functions.
- The SQLite index stores face embeddings as `float32` BLOBs, with image and face metadata. Scanning computes a SHA-256 content hash and skips exact duplicate images before DeepFace extraction; multi-directory scans commit after each directory.
- See [README.md](README.md) for installation, command usage, score semantics, and database details. Update it when user-facing behavior or commands change.

## Environment and dependencies

- Use the repository-local `.venv` through `uv`; do not use system Python, another virtual environment, or bare `pip`.
- Python is constrained to `>=3.12,<3.13` to match the current DeepFace/TensorFlow dependencies.
- Sync the project and development dependencies with `uv sync --dev`.
- Add or remove dependencies with `uv add`, `uv add --dev`, or `uv remove`; update `pyproject.toml` rather than editing `uv.lock` by hand.

## Build and test

Run commands from the repository root:

```bash
uv sync --dev
uv run pytest -q
uv run doppelsort --help
```

Tests live in `tests/`. Use pytest and Click's `CliRunner` for CLI behavior; use temporary SQLite databases and monkeypatched DeepFace calls so tests do not depend on model downloads or face-recognition inference.

## Conventions

- Keep implementation in the existing `src/doppelsort/` package and follow its type annotations and `Path`-based file handling.
- Preserve SQLite compatibility with existing index files when changing schemas or queries.
- Keep command options, defaults, and output aligned with the README and Click help.
