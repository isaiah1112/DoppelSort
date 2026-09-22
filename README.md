# DoppelSort

<p align="center">
  <img src="logo.png" alt="DoppelSort logo" width="500" />
</p>

DoppelSort is a Python CLI for scanning a directory of portraits, indexing facial embeddings in SQLite, and ranking the closest matches to a source portrait image.

## Installation

This project uses uv for environment and dependency management.

```bash
uv sync --dev
uv run doppelsort --help
```

If you want to install it as a normal package entry point in the active environment:

```bash
uv pip install -e .
doppelsort --help
```

## CLI overview

```bash
uv run doppelsort --help
```

Available commands:

- `scan`: scan a directory of portrait images and build a SQLite face index
- `match`: compare a source image against a SQLite face index and return the closest matches

## Build the face index

```bash
uv run doppelsort scan ./targets --db-path ./faces.db --overwrite
```

Options:

- `--db-path`: SQLite database location
- `--model-name`: face embedding model (default: `ArcFace`)
- `--detector-backend`: face detector backend
- `--overwrite / --append`: replace or append to the existing index

## Match a source portrait against the index

```bash
uv run doppelsort match ./source.jpg --db-path ./faces.db --top-n 10
uv run doppelsort match ./source.jpg --db-path ./faces.db --json
```

This workflow uses DeepFace to extract facial embeddings from the source image and stored target faces, then ranks them by Euclidean distance and similarity score.

## Notes

- The SQLite database stores compact `float32` embeddings in binary form for efficient comparison.
- The index records each face with:
  - `id`
  - `image_path`
  - `bbox`
  - `embedding`
  - `embedding_norm`
  - `timestamp`
- This project targets Python 3.12 for compatibility with the current DeepFace/TensorFlow stack.
