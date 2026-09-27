# AGENTS.md

This repository is managed with `uv` and uses a project-local `.venv` virtual environment. Follow these rules for all work in this repo.

## Project overview

- Python package: `doppelsort`
- Source layout: `src/doppelsort/`
- CLI entry point: `doppelsort`
- Package metadata and dependencies are defined in `pyproject.toml`.

## Required environment and tooling

1. Use `uv` for all dependency, environment, and command execution tasks.
2. Use the repository-local `.venv` as the active environment for local development and verification.
3. Do not rely on the system Python, global `pip`, or a different virtualenv.
4. Prefer `uv run ...` for commands that should execute inside the project environment.
5. Prefer `source .venv/bin/activate` when a shell command needs an activated local environment.

## Standard commands

Install or sync environment:

```bash
uv sync
```

Run Python commands in the project environment:

```bash
uv run python ...
```

Run tests:

```bash
uv run pytest -q
```

Run the CLI:

```bash
uv run doppelsort --help
```

If you need the interpreter path directly:

```bash
.venv/bin/python
```

## Dependency management rules

- Add or update dependencies via `uv`, not `pip`.
- Edit `pyproject.toml` for project dependencies and tool config.
- Use `uv add <package>` for new runtime dependencies.
- Use `uv add --dev <package>` for development dependencies.
- Use `uv remove <package>` when removing dependencies.
- Keep the project environment aligned with the repo by running `uv sync` after dependency changes.

## Guardrails

- Never install packages with bare `pip install ...` in this repository.
- Never execute tests or scripts with plain `python` when the repo-managed environment is available.
- Never create or use a different venv name for this project; the expected environment is `.venv` at the repo root.
- For reproducible work, prefer `uv run` so the command is executed with the correct `.venv` settings automatically.

## Local development expectations

This project is intended to be run from the repo root. When you need to inspect or validate behavior, use the project environment created by `uv sync` and the `.venv` interpreter rather than a globally installed Python.
