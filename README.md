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

- `scan`: scan one or more directories of portrait images and build or extend a SQLite face index
- `match`: compare a source image against a SQLite face index and return the closest matches
- `db info`: inspect the SQLite face index and summarize row counts and duplicate data
- `db optimize`: remove duplicate face rows while keeping the earliest record for each unique image + embedding pair

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

## Inspect and maintain the database

```bash
uv run doppelsort db info faces.db
uv run doppelsort db optimize faces.db
```

`db info` reports summary stats such as:

- `rows`: total face entries in the SQLite database
- `unique_images`: number of distinct image paths
- `unique_embeddings`: number of distinct embedding vectors
- `duplicate_rows`: number of rows that share the same image path and embedding
- `first_seen` / `last_seen`: earliest and latest ingestion timestamps

The difference between `unique_images` and `unique_embeddings` matters because a single image can produce multiple detected faces. If an image contains multiple faces, or if the same image was scanned more than once with different detections, `unique_embeddings` may be greater than `unique_images`.

`db optimize` removes redundant rows while keeping the earliest record for each unique image + embedding pair. This is useful after repeated scans or multi-face indexing where the same face was inserted more than once.

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
