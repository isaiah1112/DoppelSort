from __future__ import annotations

import hashlib
import json as json_module
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import click
import numpy as np

os.environ.setdefault("DEEPFACE_LOG_LEVEL", str(logging.ERROR))
from deepface import DeepFace

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _warn_if_cpu_only() -> None:
    try:
        import tensorflow as tf
    except Exception:
        return

    try:
        gpus = tf.config.list_physical_devices("GPU")
    except (AttributeError, RuntimeError, ValueError):
        gpus = []

    if not gpus:
        click.echo(
            "Warning: TensorFlow is running on CPU because no GPU device was detected. Performance may be slower.",
            err=True,
        )


def _list_candidate_images(target_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in target_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: str(path),
    )


def _extract_face_entries(
    image_path: Path,
    model_name: str,
    detector_backend: str,
) -> list[dict[str, Any]]:
    result = DeepFace.represent(
        img_path=str(image_path),
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=True,
        align=True,
    )

    if isinstance(result, dict):
        payload = [result]
    elif isinstance(result, list) and result and isinstance(result[0], dict):
        payload = result
    elif isinstance(result, list):
        payload = [{"embedding": result}]
    else:
        payload = []

    faces: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        embedding = item.get("embedding")
        if embedding is None:
            continue

        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue

        bbox = item.get("facial_area") or item.get("bbox") or {"x": 0, "y": 0, "w": 0, "h": 0}
        faces.append({"embedding": arr.astype(np.float32), "bbox": bbox})

    return faces


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    array = np.asarray(embedding, dtype=np.float32)
    return array.tobytes()


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _compute_content_hash(image_path: Path) -> str:
    digest = hashlib.sha256()
    with image_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_perceptual_hash(image_path: Path) -> str | None:
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(image_path) as image:
            image = image.convert("L").resize((8, 8), getattr(Image, "Resampling", Image).LANCZOS)
            pixels = np.asarray(image, dtype=np.float32)
    except Exception:
        return None

    mean_value = float(pixels.mean())
    bits = (pixels > mean_value).astype(np.uint8).flatten()
    hash_value = int("".join(str(bit) for bit in bits), 2)
    return f"{hash_value:064x}"


def _ensure_image_schema(conn: sqlite3.Connection) -> None:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "images" not in tables:
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

    image_columns = {row[1] for row in conn.execute("PRAGMA table_info(images)").fetchall()}
    if "content_hash" not in image_columns:
        conn.execute("ALTER TABLE images ADD COLUMN content_hash TEXT")
    if "perceptual_hash" not in image_columns:
        conn.execute("ALTER TABLE images ADD COLUMN perceptual_hash TEXT")
    if "timestamp" not in image_columns:
        conn.execute("ALTER TABLE images ADD COLUMN timestamp INTEGER NOT NULL DEFAULT 0")
    if "file_size" not in image_columns:
        conn.execute("ALTER TABLE images ADD COLUMN file_size INTEGER")
    if "mtime" not in image_columns:
        conn.execute("ALTER TABLE images ADD COLUMN mtime INTEGER")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_content_hash ON images(content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_perceptual_hash ON images(perceptual_hash)")


def _ensure_face_schema(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(faces)").fetchall()}
    if "content_hash" not in columns:
        conn.execute("ALTER TABLE faces ADD COLUMN content_hash TEXT")
    if "perceptual_hash" not in columns:
        conn.execute("ALTER TABLE faces ADD COLUMN perceptual_hash TEXT")
    if "image_id" not in columns:
        conn.execute("ALTER TABLE faces ADD COLUMN image_id INTEGER")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_image_path ON faces(image_path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_content_hash ON faces(content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_perceptual_hash ON faces(perceptual_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id)")


def _create_face_index_db(db_path: Path, *, overwrite: bool = False) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL,
            bbox TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_norm REAL,
            timestamp INTEGER NOT NULL,
            content_hash TEXT,
            perceptual_hash TEXT,
            image_id INTEGER
        )
        """
    )
    _ensure_image_schema(conn)
    _ensure_face_schema(conn)
    if overwrite:
        conn.execute("DELETE FROM faces")
        conn.execute("DELETE FROM images")
    return conn


def _index_target_directory_into_conn(
    conn: sqlite3.Connection,
    target_dir: Path,
    *,
    model_name: str,
    detector_backend: str,
) -> int:
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_dir}")

    candidates = _list_candidate_images(target_dir)
    if not candidates:
        raise ValueError(f"No supported image files were found in {target_dir}")

    inserted_rows = 0
    pending_face_rows: list[tuple[str, str, bytes, float, int, str | None, str | None, int | None]] = []
    with click.progressbar(candidates, label=f"Indexing {target_dir.name}") as progress:
        for image_path in progress:
            content_hash = _compute_content_hash(image_path)
            perceptual_hash = _compute_perceptual_hash(image_path)
            file_stat = image_path.stat()

            existing_image = conn.execute(
                "SELECT id FROM images WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            ).fetchone()
            if content_hash and existing_image is not None:
                continue

            conn.execute(
                """
                INSERT INTO images (image_path, content_hash, perceptual_hash, timestamp, file_size, mtime)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    perceptual_hash = excluded.perceptual_hash,
                    timestamp = excluded.timestamp,
                    file_size = excluded.file_size,
                    mtime = excluded.mtime
                """,
                (
                    str(image_path),
                    content_hash,
                    perceptual_hash,
                    int(time.time()),
                    file_stat.st_size,
                    int(file_stat.st_mtime),
                ),
            )
            image_row = conn.execute(
                "SELECT id FROM images WHERE image_path = ? LIMIT 1",
                (str(image_path),),
            ).fetchone()
            image_id_value = int(image_row[0]) if image_row is not None else None

            try:
                faces = _extract_face_entries(image_path, model_name=model_name, detector_backend=detector_backend)
            except Exception:
                continue

            for face in faces:
                embedding = face["embedding"]
                bbox = face["bbox"]
                pending_face_rows.append(
                    (
                        str(image_path),
                        json_module.dumps(bbox, sort_keys=True),
                        _serialize_embedding(embedding),
                        float(np.linalg.norm(embedding)),
                        int(time.time()),
                        content_hash,
                        perceptual_hash,
                        image_id_value,
                    )
                )

            if len(pending_face_rows) >= 128:
                conn.executemany(
                    """
                    INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp, content_hash, perceptual_hash, image_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    pending_face_rows,
                )
                inserted_rows += len(pending_face_rows)
                pending_face_rows.clear()

    if pending_face_rows:
        conn.executemany(
            """
            INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp, content_hash, perceptual_hash, image_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            pending_face_rows,
        )
        inserted_rows += len(pending_face_rows)
        pending_face_rows.clear()

    return inserted_rows


def index_target_directory(
    target_dir: Path,
    db_path: Path,
    model_name: str = "ArcFace",
    detector_backend: str = "retinaface",
    overwrite: bool = True,
) -> int:
    conn = _create_face_index_db(db_path, overwrite=overwrite)
    try:
        inserted_rows = _index_target_directory_into_conn(
            conn,
            target_dir,
            model_name=model_name,
            detector_backend=detector_backend,
        )
        conn.commit()
        return inserted_rows
    finally:
        conn.close()


def index_target_directories(
    target_dirs: list[Path],
    db_path: Path,
    model_name: str = "ArcFace",
    detector_backend: str = "retinaface",
    overwrite: bool = True,
) -> int:
    if not target_dirs:
        raise ValueError("At least one target directory is required.")

    conn = _create_face_index_db(db_path, overwrite=overwrite)
    inserted_rows = 0
    try:
        for target_dir in target_dirs:
            inserted_rows += _index_target_directory_into_conn(
                conn,
                target_dir,
                model_name=model_name,
                detector_backend=detector_backend,
            )
            conn.commit()
        return inserted_rows
    finally:
        conn.close()


def _load_db_faces(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        raise FileNotFoundError(f"Face database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, image_path, bbox, embedding, embedding_norm FROM faces"
        ).fetchall()
    finally:
        conn.close()

    faces: list[dict[str, Any]] = []
    for row in rows:
        record_id, image_path, bbox_json, embedding_blob, embedding_norm = row
        if not embedding_blob:
            continue

        arr = _deserialize_embedding(embedding_blob)
        if arr.size == 0:
            continue

        faces.append(
            {
                "id": record_id,
                "path": image_path,
                "bbox": json_module.loads(bbox_json) if bbox_json else {},
                "embedding": arr.astype(np.float32),
                "embedding_norm": float(embedding_norm) if embedding_norm is not None else float(np.linalg.norm(arr)),
            }
        )

    return faces


def _to_similarity_percent(similarity: float) -> float:
    return round(max(0.0, min(100.0, float(similarity) * 100.0)), 2)


def _to_distance_percent(similarity: float) -> float:
    return round(max(0.0, min(100.0, (1.0 - float(similarity)) * 100.0)), 2)


def _score_embeddings(source_embedding: np.ndarray, candidate_embedding: np.ndarray) -> tuple[float, float]:
    distance = float(np.linalg.norm(source_embedding - candidate_embedding))
    similarity = 1.0 / (1.0 + distance)
    return distance, similarity


def _sort_match_rows(rows: list[dict[str, object]], sort_by: str) -> list[dict[str, object]]:
    normalized_sort = sort_by.lower()
    if normalized_sort not in {"similarity", "distance"}:
        raise ValueError("sort_by must be either 'similarity' or 'distance'")

    if normalized_sort == "distance":
        # Lower distance is better when ranking by distance.
        return sorted(rows, key=lambda item: (float(item["_raw_distance"]), -float(item["_raw_similarity"])))

    return sorted(rows, key=lambda item: (-float(item["_raw_similarity"]), float(item["_raw_distance"])))


def match_source_to_db(
    source_image: Path,
    db_path: Path,
    top_n: int = 5,
    model_name: str = "ArcFace",
    detector_backend: str = "retinaface",
    sort_by: str = "similarity",
) -> list[dict[str, object]]:
    if not source_image.exists():
        raise FileNotFoundError(f"Source image not found: {source_image}")

    source_faces = _extract_face_entries(
        image_path=source_image,
        model_name=model_name,
        detector_backend=detector_backend,
    )
    if not source_faces:
        raise ValueError(
            "No valid face was detected in the source image. Check that the source image contains a detectable portrait."
        )

    source_embedding = np.mean(np.stack([face["embedding"] for face in source_faces]), axis=0).astype(np.float32)
    db_faces = _load_db_faces(db_path)
    if not db_faces:
        raise ValueError(f"No face embeddings were found in the SQLite database: {db_path}")

    scored_matches: list[dict[str, object]] = []
    for record in db_faces:
        candidate_embedding = record["embedding"]
        distance, similarity = _score_embeddings(source_embedding, candidate_embedding)
        scored_matches.append(
            {
                "id": record["id"],
                "path": record["path"],
                "bbox": record["bbox"],
                "distance": _to_distance_percent(similarity),
                "similarity": _to_similarity_percent(similarity),
                "_raw_distance": distance,
                "_raw_similarity": similarity,
            }
        )

    scored_matches = _sort_match_rows(scored_matches, sort_by)

    for item in scored_matches:
        item.pop("_raw_distance", None)
        item.pop("_raw_similarity", None)
    return scored_matches[:top_n]


def find_top_matches(
    source_image: Path,
    target_dir: Path,
    top_n: int = 5,
    model_name: str = "ArcFace",
    detector_backend: str = "retinaface",
    sort_by: str = "similarity",
) -> list[dict[str, object]]:
    if not source_image.exists():
        raise FileNotFoundError(f"Source image not found: {source_image}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_dir}")

    candidates = _list_candidate_images(target_dir)
    if not candidates:
        raise ValueError(f"No supported image files were found in {target_dir}")

    matches: list[dict[str, object]] = []
    for target_image in candidates:
        try:
            result = DeepFace.verify(
                img1_path=str(source_image),
                img2_path=str(target_image),
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=True,
            )
        except Exception:
            continue

        distance = float(result.get("distance", 1.0))
        similarity = max(0.0, 1.0 - distance)
        matches.append(
            {
                "path": str(target_image),
                "distance": _to_distance_percent(similarity),
                "similarity": _to_similarity_percent(similarity),
                "_raw_distance": distance,
                "_raw_similarity": similarity,
            }
        )

    if not matches:
        raise ValueError(
            "No valid face matches were found. Check that the source and target images contain detectable faces."
        )

    matches = _sort_match_rows(matches, sort_by)

    for item in matches:
        item.pop("_raw_distance", None)
        item.pop("_raw_similarity", None)
    return matches[:top_n]


def compare_images(
    source_image: Path,
    target_image: Path,
    model_name: str = "ArcFace",
    detector_backend: str = "retinaface",
    sort_by: str = "similarity",
) -> list[dict[str, object]]:
    if not source_image.exists():
        raise FileNotFoundError(f"Source image not found: {source_image}")
    if not target_image.exists():
        raise FileNotFoundError(f"Target image not found: {target_image}")

    source_faces = _extract_face_entries(
        image_path=source_image,
        model_name=model_name,
        detector_backend=detector_backend,
    )
    target_faces = _extract_face_entries(
        image_path=target_image,
        model_name=model_name,
        detector_backend=detector_backend,
    )
    if not source_faces or not target_faces:
        raise ValueError(
            "No valid faces were detected in the source or target image. Check that both images contain detectable portraits."
        )

    source_embedding = np.mean(np.stack([face["embedding"] for face in source_faces]), axis=0).astype(np.float32)
    target_embedding = np.mean(np.stack([face["embedding"] for face in target_faces]), axis=0).astype(np.float32)
    distance, similarity = _score_embeddings(source_embedding, target_embedding)

    result: list[dict[str, object]] = [
        {
            "path": str(target_image),
            "distance": _to_distance_percent(similarity),
            "similarity": _to_similarity_percent(similarity),
            "_raw_distance": distance,
            "_raw_similarity": similarity,
        }
    ]

    result = _sort_match_rows(result, sort_by)
    for item in result:
        item.pop("_raw_distance", None)
        item.pop("_raw_similarity", None)
    return result


def get_db_summary(db_path: Path) -> dict[str, int | str | None]:
    if not db_path.exists():
        raise FileNotFoundError(f"Face database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        face_columns = {row[1] for row in conn.execute("PRAGMA table_info(faces)").fetchall()}
        image_columns = {row[1] for row in conn.execute("PRAGMA table_info(images)").fetchall()} if "images" in {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        } else set()

        row_count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        unique_images = conn.execute("SELECT COUNT(DISTINCT image_path) FROM faces").fetchone()[0]
        unique_embeddings = conn.execute("SELECT COUNT(DISTINCT embedding) FROM faces").fetchone()[0]
        duplicate_rows = conn.execute(
            "SELECT COUNT(*) FROM (SELECT image_path, embedding, COUNT(*) AS dup_count FROM faces GROUP BY image_path, embedding HAVING dup_count > 1)"
        ).fetchone()[0]
        unique_content_hashes = None
        duplicate_content_hashes = None
        if "content_hash" in face_columns:
            unique_content_hashes = conn.execute(
                "SELECT COUNT(DISTINCT content_hash) FROM faces WHERE content_hash IS NOT NULL"
            ).fetchone()[0]
            duplicate_content_hashes = conn.execute(
                "SELECT COUNT(*) FROM (SELECT content_hash, COUNT(*) AS dup_count FROM faces WHERE content_hash IS NOT NULL GROUP BY content_hash HAVING dup_count > 1)"
            ).fetchone()[0]
        if "content_hash" in image_columns:
            unique_content_hashes = conn.execute(
                "SELECT COUNT(DISTINCT content_hash) FROM images WHERE content_hash IS NOT NULL"
            ).fetchone()[0]
            duplicate_content_hashes = conn.execute(
                "SELECT COUNT(*) FROM (SELECT content_hash, COUNT(*) AS dup_count FROM images WHERE content_hash IS NOT NULL GROUP BY content_hash HAVING dup_count > 1)"
            ).fetchone()[0]
        timestamp_min = conn.execute("SELECT MIN(timestamp) FROM faces").fetchone()[0]
        timestamp_max = conn.execute("SELECT MAX(timestamp) FROM faces").fetchone()[0]
    finally:
        conn.close()

    return {
        "rows": int(row_count),
        "unique_images": int(unique_images),
        "unique_embeddings": int(unique_embeddings),
        "duplicate_rows": int(duplicate_rows),
        "unique_content_hashes": int(unique_content_hashes) if unique_content_hashes is not None else None,
        "duplicate_content_hashes": int(duplicate_content_hashes) if duplicate_content_hashes is not None else None,
        "first_seen": int(timestamp_min) if timestamp_min is not None else None,
        "last_seen": int(timestamp_max) if timestamp_max is not None else None,
    }


def optimize_db(db_path: Path) -> dict[str, int | str]:
    if not db_path.exists():
        raise FileNotFoundError(f"Face database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(faces)").fetchall()}
        before_rows = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        dedupe_key = "content_hash" if "content_hash" in columns else "image_path"
        duplicate_rows = conn.execute(
            f"SELECT COUNT(*) FROM (SELECT id FROM faces WHERE id NOT IN (SELECT MIN(id) FROM faces GROUP BY {dedupe_key}, embedding))"
        ).fetchone()[0]

        conn.execute(
            f"DELETE FROM faces WHERE id NOT IN (SELECT MIN(id) FROM faces GROUP BY {dedupe_key}, embedding)"
        )
        conn.commit()
        after_rows = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    finally:
        conn.close()

    return {
        "removed_rows": int(before_rows - after_rows),
        "duplicate_rows": int(duplicate_rows),
        "rows_after": int(after_rows),
    }


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Match a source portrait against indexed candidate faces or build a face index from a target directory.",
)
def main() -> None:
    """CLI entry point for indexing and matching portraits."""
    return


@main.group("db")
def db() -> None:
    """Inspect or maintain the SQLite face index."""
    return


@db.command("info")
@click.argument("db_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def db_info(db_path: Path) -> None:
    """Print summary information about the face database."""
    try:
        summary = get_db_summary(db_path)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Database: {db_path}")
    click.echo(f"rows: {summary['rows']}")
    click.echo(f"unique_images: {summary['unique_images']}")
    click.echo(f"unique_embeddings: {summary['unique_embeddings']}")
    click.echo(f"duplicate_rows: {summary['duplicate_rows']}")
    if summary["unique_content_hashes"] is not None:
        click.echo(f"unique_content_hashes: {summary['unique_content_hashes']}")
    if summary["duplicate_content_hashes"] is not None:
        click.echo(f"duplicate_content_hashes: {summary['duplicate_content_hashes']}")
    if summary["first_seen"] is not None:
        click.echo(f"first_seen: {summary['first_seen']}")
    if summary["last_seen"] is not None:
        click.echo(f"last_seen: {summary['last_seen']}")


@db.command("optimize")
@click.argument("db_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def db_optimize(db_path: Path) -> None:
    """Remove duplicate face rows while keeping the earliest record for each image + embedding."""
    try:
        result = optimize_db(db_path)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Removed {result['removed_rows']} duplicate rows from {db_path}")
    click.echo(f"Rows after optimization: {result['rows_after']}")


@main.command("scan")
@click.argument("target_dirs", nargs=-1, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--db-path",
    default="faces.db",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination SQLite database path for the face index.",
)
@click.option(
    "--model-name",
    default="ArcFace",
    show_default=True,
    help="DeepFace embedding model to use while indexing target portraits.",
)
@click.option(
    "--detector-backend",
    default="retinaface",
    show_default=True,
    type=click.Choice(["retinaface", "mtcnn", "opencv", "ssd", "dlib", "mediapipe"], case_sensitive=False),
    help="Detector backend used to find faces in the target image set.",
)
@click.option(
    "--overwrite/--append",
    default=True,
    show_default=True,
    help="Clear the existing index before scanning or append new faces to the database.",
)
def scan(
    target_dirs: tuple[Path, ...],
    db_path: Path,
    model_name: str,
    detector_backend: str,
    overwrite: bool,
) -> None:
    """Scan one or more directories of portraits and create or extend a SQLite face index."""
    _warn_if_cpu_only()
    if not target_dirs:
        raise click.UsageError("At least one target directory is required.")

    try:
        inserted = index_target_directories(
            target_dirs=list(target_dirs),
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            overwrite=overwrite,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Indexed {inserted} faces into {db_path}")


@main.command("compare")
@click.argument("source_image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("target_image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--model-name",
    default="ArcFace",
    show_default=True,
    help="DeepFace embedding model to use for the comparison.",
)
@click.option(
    "--detector-backend",
    default="retinaface",
    show_default=True,
    type=click.Choice(["retinaface", "mtcnn", "opencv", "ssd", "dlib", "mediapipe"], case_sensitive=False),
    help="Detector backend used to find faces in the source and target images.",
)
@click.option(
    "--by-similarity/--by-distance",
    default=True,
    help="Rank the comparison by the highest similarity score or the lowest distance value (lower distance is better).",
)
@click.option(
    "--json",
    is_flag=True,
    default=False,
    help="Emit the result as JSON instead of a text row.",
)
def compare(
    source_image: Path,
    target_image: Path,
    model_name: str,
    detector_backend: str,
    by_similarity: bool,
    json: bool,
) -> None:
    """Compare two portraits and report their similarity and distance scores."""
    _warn_if_cpu_only()
    sort_by = "similarity" if by_similarity else "distance"
    try:
        results = compare_images(
            source_image=source_image,
            target_image=target_image,
            model_name=model_name,
            detector_backend=detector_backend,
            sort_by=sort_by,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if json:
        click.echo(json_module.dumps(results, indent=2))
        return

    for idx, result in enumerate(results, start=1):
        click.echo(f"{idx}. {result['path']} | distance={result['distance']}% | similarity={result['similarity']}%")


@main.command("match")
@click.argument("source_image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--db-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="SQLite face index created by the scan command.",
)
@click.option(
    "--top-n",
    default=5,
    show_default=True,
    type=click.IntRange(1),
    help="Return the N closest matches from the SQLite database.",
)
@click.option(
    "--model-name",
    default="ArcFace",
    show_default=True,
    help="DeepFace embedding model to use for the source image comparison.",
)
@click.option(
    "--detector-backend",
    default="retinaface",
    show_default=True,
    type=click.Choice(["retinaface", "mtcnn", "opencv", "ssd", "dlib", "mediapipe"], case_sensitive=False),
    help="Detector backend used to find the face in the source image.",
)
@click.option(
    "--by-similarity/--by-distance",
    default=True,
    help="Rank matches by the highest similarity score or the lowest distance value (lower distance is better).",
)
@click.option(
    "--json",
    is_flag=True,
    default=False,
    help="Emit the results as JSON instead of a text table.",
)
def match(
    source_image: Path,
    db_path: Path,
    top_n: int,
    model_name: str,
    detector_backend: str,
    by_similarity: bool,
    json: bool,
) -> None:
    """Match a source portrait against faces already indexed in SQLite."""
    _warn_if_cpu_only()
    sort_by = "similarity" if by_similarity else "distance"
    try:
        results = match_source_to_db(
            source_image=source_image,
            db_path=db_path,
            top_n=top_n,
            model_name=model_name,
            detector_backend=detector_backend,
            sort_by=sort_by,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if json:
        click.echo(json_module.dumps(results, indent=2))
        return

    for idx, result in enumerate(results, start=1):
        click.echo(f"{idx}. {result['path']} | distance={result['distance']}% | similarity={result['similarity']}%")


if __name__ == "__main__":
    main()
