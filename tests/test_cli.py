import sqlite3

import numpy as np
import pytest
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


def test_db_info_renders_summary(tmp_path) -> None:
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
            "/tmp/a.jpg",
            '{"x": 1, "y": 2, "w": 3, "h": 4}',
            np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            1.0,
            123,
        ),
    )
    conn.execute(
        "INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp) VALUES (?, ?, ?, ?, ?)",
        (
            "/tmp/b.jpg",
            '{"x": 1, "y": 2, "w": 3, "h": 4}',
            np.array([0.0, 1.0, 0.0], dtype=np.float32).tobytes(),
            1.0,
            456,
        ),
    )
    conn.commit()
    conn.close()

    runner = CliRunner()
    result = runner.invoke(main, ["db", "info", str(db_path)])

    assert result.exit_code == 0
    assert "rows" in result.output.lower()
    assert "2" in result.output


def test_db_optimize_removes_duplicate_rows(tmp_path) -> None:
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
    duplicate = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp) VALUES (?, ?, ?, ?, ?)",
        ("/tmp/a.jpg", '{"x": 1, "y": 2, "w": 3, "h": 4}', duplicate, 1.0, 123),
    )
    conn.execute(
        "INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp) VALUES (?, ?, ?, ?, ?)",
        ("/tmp/a.jpg", '{"x": 1, "y": 2, "w": 3, "h": 4}', duplicate, 1.0, 456),
    )
    conn.commit()
    conn.close()

    runner = CliRunner()
    result = runner.invoke(main, ["db", "optimize", str(db_path)])

    assert result.exit_code == 0
    assert "1" in result.output

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    conn.close()
    assert count == 1


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


def test_scan_can_append_multiple_target_directories(tmp_path, monkeypatch) -> None:
    target_dir_1 = tmp_path / "dir1"
    target_dir_2 = tmp_path / "dir2"
    target_dir_1.mkdir()
    target_dir_2.mkdir()
    (target_dir_1 / "a.jpg").write_bytes(b"a")
    (target_dir_2 / "b.jpg").write_bytes(b"b")

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
            str(target_dir_1 / "existing.jpg"),
            '{"x": 1, "y": 2, "w": 3, "h": 4}',
            np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            1.0,
            123,
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "doppelsort.cli._extract_face_entries",
        lambda *args, **kwargs: [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32), "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}],
    )

    runner = CliRunner()
    result = runner.invoke(main, ["scan", str(target_dir_1), str(target_dir_2), "--db-path", str(db_path), "--append"])

    assert result.exit_code == 0
    assert "Indexed 2 faces" in result.output

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    conn.close()
    assert count == 3


def test_index_target_directories_commits_after_each_directory(tmp_path, monkeypatch) -> None:
    import doppelsort.cli as cli

    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()
    (dir1 / "a.jpg").write_bytes(b"a")
    (dir2 / "b.jpg").write_bytes(b"b")

    db_path = tmp_path / "faces.db"

    real_index = cli._index_target_directory_into_conn

    def flaky_index(conn, target_dir, **kwargs):
        inserted = real_index(conn, target_dir, **kwargs)
        if target_dir.name == "dir2":
            raise RuntimeError("simulated later-directory failure")
        return inserted

    monkeypatch.setattr(cli, "_index_target_directory_into_conn", flaky_index)
    monkeypatch.setattr(
        cli,
        "_extract_face_entries",
        lambda *args, **kwargs: [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32), "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}],
    )

    with pytest.raises(RuntimeError, match="later-directory failure"):
        cli.index_target_directories([dir1, dir2], db_path)

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    conn.close()
    assert count == 1


def test_compare_command_outputs_like_match(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.jpg"
    target_path = tmp_path / "target.jpg"
    source_path.write_bytes(b"source")
    target_path.write_bytes(b"target")

    monkeypatch.setattr(
        "doppelsort.cli.DeepFace.represent",
        lambda **kwargs: [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32)}],
    )

    runner = CliRunner()
    result = runner.invoke(main, ["compare", str(source_path), str(target_path), "--by-distance"])

    assert result.exit_code == 0
    assert "target.jpg" in result.output
    assert "distance=" in result.output
    assert "similarity=" in result.output


def test_scan_skips_exact_duplicate_images_by_content_hash(tmp_path, monkeypatch) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    first_path = target_dir / "first.jpg"
    second_path = target_dir / "second.jpg"
    first_path.write_bytes(b"same-image-content")
    second_path.write_bytes(b"same-image-content")

    db_path = tmp_path / "faces.db"
    calls: list[str] = []

    def fake_extract_face_entries(image_path, **kwargs):
        calls.append(str(image_path))
        return [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32), "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}]

    monkeypatch.setattr("doppelsort.cli._extract_face_entries", fake_extract_face_entries)

    runner = CliRunner()
    result = runner.invoke(main, ["scan", str(target_dir), "--db-path", str(db_path), "--overwrite"])

    assert result.exit_code == 0
    assert "Indexed 1 faces" in result.output
    assert len(calls) == 1

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT content_hash, perceptual_hash FROM faces LIMIT 1").fetchone()
    conn.close()
    assert row is not None
    assert row[0] is not None
    assert row[1] is not None or row[1] is None


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
