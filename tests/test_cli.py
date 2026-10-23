import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from click.testing import CliRunner

from doppelsort.cli import find_top_matches, main


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


def test_db_info_summarizes_current_image_hash_metadata(tmp_path) -> None:
    db_path = tmp_path / "faces.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE images (
            id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL UNIQUE,
            content_hash TEXT,
            perceptual_hash TEXT,
            timestamp INTEGER NOT NULL,
            file_size INTEGER,
            mtime INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE faces (
            id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL,
            bbox TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_norm REAL,
            timestamp INTEGER NOT NULL,
            content_hash TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO images (image_path, content_hash, timestamp) VALUES (?, ?, ?)",
        [
            ("/photos/a.jpg", "same-hash", 10),
            ("/photos/b.jpg", "same-hash", 20),
            ("/photos/c.jpg", "other-hash", 30),
        ],
    )
    embedding_a = np.array([1.0, 0.0], dtype=np.float32).tobytes()
    embedding_b = np.array([0.0, 1.0], dtype=np.float32).tobytes()
    conn.executemany(
        "INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp, content_hash) VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("/photos/a.jpg", "{}", embedding_a, 1.0, 10, "same-hash"),
            ("/photos/a.jpg", "{}", embedding_a, 1.0, 20, "same-hash"),
            ("/photos/b.jpg", "{}", embedding_a, 1.0, 30, "same-hash"),
            ("/photos/c.jpg", "{}", embedding_a, 1.0, 40, "other-hash"),
            ("/photos/c.jpg", "{}", embedding_b, 1.0, 50, "other-hash"),
        ],
    )
    conn.commit()
    conn.close()

    runner = CliRunner()
    result = runner.invoke(main, ["db", "info", str(db_path)])

    assert result.exit_code == 0
    assert "rows: 5" in result.output
    assert "unique_images: 3" in result.output
    assert "unique_embeddings: 2" in result.output
    assert "duplicate_rows: 1" in result.output
    assert "unique_content_hashes: 2" in result.output
    assert "duplicate_content_hashes: 1" in result.output
    assert "first_seen: 10" in result.output
    assert "last_seen: 50" in result.output


def test_db_optimize_deduplicates_by_content_hash_for_current_index(tmp_path) -> None:
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
            timestamp INTEGER NOT NULL,
            content_hash TEXT
        )
        """
    )
    embedding_a = np.array([1.0, 0.0], dtype=np.float32).tobytes()
    embedding_b = np.array([0.0, 1.0], dtype=np.float32).tobytes()
    conn.executemany(
        "INSERT INTO faces (id, image_path, bbox, embedding, embedding_norm, timestamp, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "/photos/a.jpg", "{}", embedding_a, 1.0, 10, "same-hash"),
            (2, "/photos/a.jpg", "{}", embedding_a, 1.0, 20, "same-hash"),
            (3, "/photos/b.jpg", "{}", embedding_a, 1.0, 30, "same-hash"),
            (4, "/photos/c.jpg", "{}", embedding_a, 1.0, 40, "other-hash"),
            (5, "/photos/c.jpg", "{}", embedding_b, 1.0, 50, "other-hash"),
        ],
    )
    conn.commit()
    conn.close()

    runner = CliRunner()
    result = runner.invoke(main, ["db", "optimize", str(db_path)])

    assert result.exit_code == 0
    assert "Removed 2 duplicate rows" in result.output
    conn = sqlite3.connect(db_path)
    remaining_ids = conn.execute("SELECT id FROM faces ORDER BY id").fetchall()
    conn.close()
    assert remaining_ids == [(1,), (4,), (5,)]


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


def test_scan_adds_missing_columns_to_legacy_images_table(tmp_path, monkeypatch) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    image_path = target_dir / "new.jpg"
    image_path.write_bytes(b"new image")

    db_path = tmp_path / "faces.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, image_path TEXT NOT NULL UNIQUE)")
    conn.execute("INSERT INTO images (image_path) VALUES (?)", (str(tmp_path / "old.jpg"),))
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
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "doppelsort.cli._extract_face_entries",
        lambda *args, **kwargs: [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32), "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}],
    )

    runner = CliRunner()
    result = runner.invoke(main, ["scan", str(target_dir), "--db-path", str(db_path), "--append"])

    assert result.exit_code == 0
    assert "Indexed 1 faces" in result.output

    conn = sqlite3.connect(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
    image_count = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    face_count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    conn.close()
    assert {"content_hash", "perceptual_hash", "timestamp", "file_size", "mtime"} <= columns
    assert image_count == 2
    assert face_count == 1


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


def test_compare_command_emits_json(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.jpg"
    target_path = tmp_path / "target.jpg"
    source_path.write_bytes(b"source")
    target_path.write_bytes(b"target")

    monkeypatch.setattr(tf.config, "list_physical_devices", lambda device_type: [object()])
    monkeypatch.setattr(
        "doppelsort.cli.DeepFace.represent",
        lambda **kwargs: [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32)}],
    )

    runner = CliRunner()
    result = runner.invoke(main, ["compare", str(source_path), str(target_path), "--json"])

    assert result.exit_code == 0
    results = json.loads(result.stdout)
    assert len(results) == 1
    assert results[0]["path"] == str(target_path)
    assert {"distance", "similarity"} <= results[0].keys()


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


def test_scan_continues_after_face_extraction_failure(tmp_path, monkeypatch) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    failed_path = target_dir / "a-failed.jpg"
    successful_path = target_dir / "b-successful.jpg"
    failed_path.write_bytes(b"failed")
    successful_path.write_bytes(b"successful")

    calls: list[str] = []

    def extract_face_entries(image_path, **kwargs):
        calls.append(str(image_path))
        if image_path == failed_path:
            raise RuntimeError("simulated extraction failure")
        return [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32), "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}]

    monkeypatch.setattr("doppelsort.cli._extract_face_entries", extract_face_entries)

    db_path = tmp_path / "faces.db"
    runner = CliRunner()
    result = runner.invoke(main, ["scan", str(target_dir), "--db-path", str(db_path)])

    assert result.exit_code == 0
    assert "Indexed 1 faces" in result.output
    assert calls == [str(failed_path), str(successful_path)]

    conn = sqlite3.connect(db_path)
    indexed_paths = conn.execute("SELECT image_path FROM faces").fetchall()
    conn.close()
    assert indexed_paths == [(str(successful_path),)]


def test_scan_inserts_faces_across_batch_threshold(tmp_path, monkeypatch) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    (target_dir / "many-faces.jpg").write_bytes(b"many-faces")
    faces = [
        {"embedding": np.array([float(index), 0.0, 0.0], dtype=np.float32), "bbox": {"x": index, "y": 0, "w": 1, "h": 1}}
        for index in range(128)
    ]
    monkeypatch.setattr("doppelsort.cli._extract_face_entries", lambda *args, **kwargs: faces)

    db_path = tmp_path / "faces.db"
    runner = CliRunner()
    result = runner.invoke(main, ["scan", str(target_dir), "--db-path", str(db_path)])

    assert result.exit_code == 0
    assert "Indexed 128 faces" in result.output

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    conn.close()
    assert count == 128


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


def test_match_command_emits_json(tmp_path, monkeypatch) -> None:
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
            str(tmp_path / "candidate.jpg"),
            '{"x": 1, "y": 2, "w": 3, "h": 4}',
            np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            1.0,
            123,
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(tf.config, "list_physical_devices", lambda device_type: [object()])
    monkeypatch.setattr(
        "doppelsort.cli.DeepFace.represent",
        lambda **kwargs: [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32)}],
    )

    runner = CliRunner()
    result = runner.invoke(main, ["match", str(source_path), "--db-path", str(db_path), "--json"])

    assert result.exit_code == 0
    results = json.loads(result.stdout)
    assert len(results) == 1
    assert results[0]["path"] == str(tmp_path / "candidate.jpg")
    assert {"id", "bbox", "distance", "similarity"} <= results[0].keys()


def test_match_reports_missing_source_face(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"source")
    db_path = tmp_path / "faces.db"
    sqlite3.connect(db_path).close()
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda device_type: [object()])
    monkeypatch.setattr("doppelsort.cli.DeepFace.represent", lambda **kwargs: [])

    runner = CliRunner()
    result = runner.invoke(main, ["match", str(source_path), "--db-path", str(db_path)])

    assert result.exit_code == 1
    assert "No valid face was detected in the source image" in result.output


def test_match_reports_empty_face_index(tmp_path, monkeypatch) -> None:
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
    conn.commit()
    conn.close()
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda device_type: [object()])
    monkeypatch.setattr(
        "doppelsort.cli.DeepFace.represent",
        lambda **kwargs: [{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32)}],
    )

    runner = CliRunner()
    result = runner.invoke(main, ["match", str(source_path), "--db-path", str(db_path)])

    assert result.exit_code == 1
    assert "No face embeddings were found in the SQLite database" in result.output


def test_find_top_matches_ranks_successful_verifications_and_limits_results(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"source")
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (target_dir / name).write_bytes(name.encode())

    def verify(*, img2_path, **kwargs):
        target_name = Path(img2_path).name
        if target_name == "a.jpg":
            return {"distance": 0.3}
        if target_name == "b.jpg":
            return {}
        raise RuntimeError("simulated verification failure")

    monkeypatch.setattr("doppelsort.cli.DeepFace.verify", verify)

    matches = find_top_matches(source_path, target_dir, top_n=2)

    assert [Path(match["path"]).name for match in matches] == ["a.jpg", "b.jpg"]
    assert matches[1]["distance"] == 100.0
    assert matches[1]["similarity"] == 0.0
    assert all("_raw_distance" not in match and "_raw_similarity" not in match for match in matches)


def test_find_top_matches_reports_invalid_inputs_and_no_matches(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"source")
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    non_directory = tmp_path / "file"
    non_directory.write_bytes(b"not a directory")

    with pytest.raises(FileNotFoundError, match="Source image not found"):
        find_top_matches(tmp_path / "missing.jpg", target_dir)
    with pytest.raises(FileNotFoundError, match="Target directory not found"):
        find_top_matches(source_path, tmp_path / "missing")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        find_top_matches(source_path, non_directory)
    with pytest.raises(ValueError, match="No supported image files"):
        find_top_matches(source_path, target_dir)

    candidate = target_dir / "candidate.jpg"
    candidate.write_bytes(b"candidate")

    def fail_verification(**kwargs):
        raise RuntimeError("simulated verification failure")

    monkeypatch.setattr(
        "doppelsort.cli.DeepFace.verify",
        fail_verification,
    )
    with pytest.raises(ValueError, match="No valid face matches"):
        find_top_matches(source_path, target_dir)
