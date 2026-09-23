import sqlite3

import numpy as np
import tensorflow as tf
from click.testing import CliRunner

from doppelsort.cli import main


def test_cli_help_renders() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "scan" in result.output.lower()
    assert "match" in result.output.lower()


def test_scan_command_help_renders() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["scan", "--help"])

    assert result.exit_code == 0
    assert "TARGET_DIR" in result.output
    assert "--db-path" in result.output


def test_warns_when_tensorflow_is_cpu_only(monkeypatch, tmp_path) -> None:
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"source")

    db_path = tmp_path / "faces.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE faces (
            id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL,
            bbox TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_norm REAL,
            timestamp INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp) VALUES (?, ?, ?, ?, ?)",
        (
            str(tmp_path / "a.jpg"),
            '{"x": 1, "y": 2, "w": 3, "h": 4}',
            np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            1.0,
            123,
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(tf.config, "list_physical_devices", lambda device_type: [])
    monkeypatch.setattr(
        "doppelsort.cli.DeepFace.represent",
        lambda **kwargs: [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32)}],
    )

    runner = CliRunner()
    result = runner.invoke(main, ["match", str(source_path), "--db-path", str(db_path)])

    assert result.exit_code == 0
    assert "Warning" in result.output
    assert "CPU" in result.output
    assert "GPU" in result.output


def test_match_supports_distance_sorting(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"source")

    db_path = tmp_path / "faces.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE faces (
            id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL,
            bbox TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_norm REAL,
            timestamp INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp) VALUES (?, ?, ?, ?, ?)",
        (
            str(tmp_path / "a.jpg"),
            '{"x": 1, "y": 2, "w": 3, "h": 4}',
            np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            1.0,
            123,
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "doppelsort.cli.DeepFace.represent",
        lambda **kwargs: [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32)}],
    )

    runner = CliRunner()
    result = runner.invoke(main, ["match", str(source_path), "--db-path", str(db_path), "--by-distance"])

    assert result.exit_code == 0
    assert "a.jpg" in result.output


def test_match_sort_by_distance_prefers_lower_score(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"source")

    db_path = tmp_path / "faces.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE faces (
            id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL,
            bbox TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_norm REAL,
            timestamp INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp) VALUES (?, ?, ?, ?, ?)",
        (
            str(tmp_path / "close.jpg"),
            '{"x": 1, "y": 2, "w": 3, "h": 4}',
            np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            1.0,
            123,
        ),
    )
    conn.execute(
        "INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp) VALUES (?, ?, ?, ?, ?)",
        (
            str(tmp_path / "far.jpg"),
            '{"x": 1, "y": 2, "w": 3, "h": 4}',
            np.array([0.0, 1.0, 0.0], dtype=np.float32).tobytes(),
            1.0,
            456,
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "doppelsort.cli.DeepFace.represent",
        lambda **kwargs: [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32)}],
    )

    runner = CliRunner()
    result = runner.invoke(main, ["match", str(source_path), "--db-path", str(db_path), "--by-distance", "--top-n", "2"])

    assert result.exit_code == 0
    assert result.output.index("close.jpg") < result.output.index("far.jpg")


def test_match_uses_sqlite_database(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"source")

    db_path = tmp_path / "faces.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE faces (
            id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL,
            bbox TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_norm REAL,
            timestamp INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp) VALUES (?, ?, ?, ?, ?)",
        (
            str(tmp_path / "a.jpg"),
            '{"x": 1, "y": 2, "w": 3, "h": 4}',
            np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            1.0,
            123,
        ),
    )
    conn.execute(
        "INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp) VALUES (?, ?, ?, ?, ?)",
        (
            str(tmp_path / "b.jpg"),
            '{"x": 1, "y": 2, "w": 3, "h": 4}',
            np.array([0.9, 0.0, 0.0], dtype=np.float32).tobytes(),
            1.0,
            456,
        ),
    )
    conn.execute(
        "INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp) VALUES (?, ?, ?, ?, ?)",
        (
            str(tmp_path / "c.jpg"),
            '{"x": 1, "y": 2, "w": 3, "h": 4}',
            np.array([0.0, 1.0, 0.0], dtype=np.float32).tobytes(),
            1.0,
            789,
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "doppelsort.cli.DeepFace.represent",
        lambda **kwargs: [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32)}],
    )

    runner = CliRunner()
    result = runner.invoke(main, ["match", str(source_path), "--db-path", str(db_path), "--top-n", "2"])

    assert result.exit_code == 0
    assert "a.jpg" in result.output
    assert "b.jpg" in result.output
    assert "c.jpg" not in result.output
    assert "%" in result.output
    assert "similarity=" in result.output
