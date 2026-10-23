# DoppelSort

<p align="center">
  <img src="logo.png" alt="DoppelSort logo" />
</p>

DoppelSort is a Python CLI for scanning portrait directories, storing face embeddings in SQLite, and comparing faces by similarity or distance.

## Installation

Requires Python 3.12. Use uv to create the project environment and install the
declared runtime dependencies, including DeepFace with its TensorFlow backend:

```bash
uv sync --dev
uv run doppelsort --help
```

You can also invoke the CLI as a Python module:

```bash
uv run python -m doppelsort --help
```

The project pins TensorFlow to the version used by its current DeepFace stack;
use `uv sync` rather than installing DeepFace or TensorFlow separately.

## CLI overview

```bash
uv run doppelsort --help
```

Available commands:

- `scan`: scan one or more directories and build or extend a SQLite face index while skipping exact duplicate images before DeepFace runs
- `compare`: compare two portrait images directly and print similarity and distance scores
- `match`: compare a source image against a SQLite face index and rank the closest matches
- `db info`: print database summary statistics including hash-based duplicate counts
- `db optimize`: remove duplicate face rows, keeping the lowest-ID row for each embedding per image content hash (or image path in older indexes)

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
- `--detector-backend`: face detection backend (default: `retinaface`; choices: `retinaface`, `mtcnn`, `opencv`, `ssd`, `dlib`, `mediapipe`)
- `--overwrite / --append`: clear the database before indexing or append to the existing database

The scan command indexes every detected face in each supported image (`.png`,
`.jpg`, `.jpeg`, `.bmp`, or `.webp`), searching each target directory
recursively. It computes a SHA-256 `content_hash` and records a lightweight
perceptual hash when possible. Files with identical content hashes are skipped
before DeepFace extraction, including copies already present in the index when
using `--append`. Perceptual hashes are stored as metadata; they are not
currently used to skip near-duplicate images.

`--overwrite` is the default and clears existing face and image rows before
scanning. Use `--append` to preserve the index. When scanning multiple
directories, each directory is committed after it completes, so earlier
directories remain saved if a later one fails.

The index stores compact `float32` embeddings as SQLite `BLOB` data, with
image and face metadata.

## SQLite schema and duplicate tracking

The database records both image metadata and face metadata.

Current image metadata includes:

- `image_path`
- `content_hash`
- `perceptual_hash`
- `timestamp`
- `file_size`
- `mtime`

Current face metadata includes:

- `id`
- `image_path`
- `bbox`
- `embedding`
- `embedding_norm`
- `timestamp`
- `content_hash`
- `perceptual_hash`
- `image_id`

The metadata supports:

- exact duplicate files (`content_hash` match)
- perceptual similarity metadata (`perceptual_hash`), which can be used for
  later near-duplicate workflows
- image records and the face detections associated with them

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

Both commands print a distance and similarity percentage from 0 to 100:

- `similarity`: higher is better
- `distance`: lower is better

These are normalized scores derived from the embedding distance, not calibrated
probabilities or identity-confidence estimates. `--by-similarity` is the
default sort order; `--by-distance` sorts from lowest distance upward. Add
`--json` to either command for a JSON array of result objects.

## Match a source portrait against the index

```bash
uv run doppelsort match ./source.jpg --db-path ./faces.db --top-n 10
uv run doppelsort match ./source.jpg --db-path ./faces.db --by-distance --json
```

This workflow extracts the source image's face embedding, loads indexed face
embeddings from SQLite, and returns at most `--top-n` candidates (default: 5).
The required `--db-path` must point to an existing index created with `scan`.

The sort behavior is:

- `--by-similarity`: highest similarity first
- `--by-distance`: lowest distance first

Text results contain the image path, distance, and similarity for each indexed
face. JSON results also include the face record's `id` and `bbox`.

## Inspect and maintain the SQLite database

```bash
uv run doppelsort db info ./faces.db
uv run doppelsort db optimize ./faces.db
```

`db info` reports summary stats such as:

- `rows`: total face entries in the SQLite database
- `unique_images`: number of distinct image paths with face rows
- `unique_embeddings`: number of distinct embedding vectors
- `duplicate_rows`: number of image-path/embedding groups containing repeated
  face rows
- `unique_content_hashes`: number of distinct image content hashes
- `duplicate_content_hashes`: number of content-hash groups appearing more
  than once
- `first_seen` / `last_seen`: earliest and latest face insertion timestamps
  (Unix timestamps)

`duplicate_rows` reports duplicate groups, not the number of excess rows in
those groups. `unique_images` counts distinct image paths with indexed face
rows, while `unique_embeddings` counts distinct embedding vectors. One image
can produce multiple face embeddings.

`db optimize` removes duplicate face rows and keeps the row with the lowest ID
for each embedding grouped by `content_hash` when that column exists, or by
`image_path` for older databases. For current indexes, identical image files
with matching embeddings therefore share a deduplication group even if their
paths differ.

## Notes

- Scan, compare, and match print a CPU warning when TensorFlow reports no GPU.
- The database uses SQLite WAL mode and adds missing image/hash metadata columns
  when opening an older index for scanning, preserving compatibility with
  earlier face tables.
