# DoppelSort
A python CLI for matching a source portrait against a directory of candidate portraits to find the closest visual matches.

## Installation

This project is managed with uv and installs as a CLI entry point.

```bash
uv sync --dev
uv run doppelsort --help
```

## Usage

```bash
uv run doppelsort path/to/source.jpg path/to/target_dir --top-n 10
uv run doppelsort path/to/source.jpg path/to/target_dir --json
```

The command uses DeepFace to detect and compare faces, then returns the closest portraits ranked by distance and similarity.
