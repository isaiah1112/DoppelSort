# DoppelSort

<p align="center">
  <img src="logo.png" alt="DoppelSort logo" width="500" />
</p>

DoppelSort is a Python CLI for scanning portrait directories, storing face embeddings in SQLite, and comparing faces by similarity or distance.

## Installation

This project uses uv for dependency and environment management.

```bash
uv sync --dev
uv run doppelsort --help
```

To install the package in the active environment:

```bash
uv pip install -e .
doppelsort --help
```

## CLI overview

```bash
uv run doppelsort --help
```

Available commands:

- `scan`: scan one or more directories and build or extend a SQLite face index
- `compare`: compare two portrait images directly and print similarity and distance scores
- `match`: compare a source image against a SQLite face index and rank the closest matches
- `db info`: print database summary statistics
- `db optimize`: remove duplicate rows while keeping the earliest record for each unique image + embedding pair

## Build a face index

```bash
uv run doppelsort scan ./targets --db-path ./faces.db --overwrite
uv run doppelsort scan ./targets_a ./targets_b --db-path ./faces.db --append
```

Arguments:

- `target_dirs`: one or more directories containing images to index

Options:

- `--db-path`: destination SQLite database path (default: `faces.db`)
- `--model-name`: DeepFace embedding model (default: `ArcFace`)
- `--detector-backend`: face detection backend
- `--overwrite / --append`: clear the database before indexing or append to the existing database

The scan command indexes every detected face in each image. It stores compact `float32` embeddings as SQLite `BLOB` data for efficient matching and retrieval.

## Compare two portraits directly

```bash
uv run doppelsort compare ./source.jpg ./target.jpg
uv run doppelsort compare ./source.jpg ./target.jpg --by-distance
uv run doppelsort compare ./source.jpg ./target.jpg --json
```

This command compares two images and prints the same text output style as `match`:

```text
1. /path/to/target.jpg | distance=12.34% | similarity=87.66%
```

The scores are normalized to a percentage from 0 to 100:

- `similarity`: higher is better
- `distance`: lower is better

## Match a source portrait against the index

```bash
uv run doppelsort match ./source.jpg --db-path ./faces.db --top-n 10
uv run doppelsort match ./source.jpg --db-path ./faces.db --by-distance --json
```

This workflow loads embeddings from the SQLite database, computes each candidate’s distance from the source embedding, and ranks candidates by either similarity or distance.

The sort behavior is:

- `--by-similarity`: highest similarity first
- `--by-distance`: lowest distance first

Scores are displayed as percentages, not raw float values.

## Inspect and maintain the SQLite database

```bash
uv run doppelsort db info ./faces.db
uv run doppelsort db optimize ./faces.db
```

`db info` reports summary stats such as:

- `rows`: total face entries in the SQLite database
- `unique_images`: number of distinct image paths
- `unique_embeddings`: number of distinct embedding vectors
- `duplicate_rows`: number of rows that share the same image path and embedding
- `first_seen` / `last_seen`: earliest and latest insertion timestamps

The difference between `unique_images` and `unique_embeddings` matters because a single image may contain multiple detected faces. If the same image is scanned repeatedly or resolves to more than one face, `unique_embeddings` can be greater than `unique_images`.

`db optimize` removes duplicate face rows while keeping the earliest record for each unique image + embedding pair. It is useful after repeated scans or multi-face indexing when the same face got inserted multiple times.

## Notes

- TensorFlow warns when no GPU is detected; the CLI prints a CPU warning in those environments.
- The index records each face with:
  - `id`
  - `image_path`
  - `bbox`
  - `embedding`
  - `embedding_norm`
  - `timestamp`
- This project targets Python 3.12 for compatibility with the current DeepFace/TensorFlow stack.
