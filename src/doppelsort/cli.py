from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
from deepface import DeepFace

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


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
            timestamp INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_image_path ON faces(image_path)")
    if overwrite:
        conn.execute("DELETE FROM faces")
    return conn


def index_target_directory(
    target_dir: Path,
    db_path: Path,
    model_name: str = "ArcFace",
    detector_backend: str = "retinaface",
    overwrite: bool = True,
) -> int:
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_dir}")

    candidates = _list_candidate_images(target_dir)
    if not candidates:
        raise ValueError(f"No supported image files were found in {target_dir}")

    conn = _create_face_index_db(db_path, overwrite=overwrite)
    inserted_rows = 0
    try:
        with click.progressbar(candidates, label="Indexing faces") as progress:
            for image_path in progress:
                try:
                    faces = _extract_face_entries(image_path, model_name=model_name, detector_backend=detector_backend)
                except Exception:
                    continue

                for face in faces:
                    embedding = face["embedding"]
                    bbox = face["bbox"]
                    embedding_blob = _serialize_embedding(embedding)
                    conn.execute(
                        """
                        INSERT INTO faces (image_path, bbox, embedding, embedding_norm, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            str(image_path),
                            json.dumps(bbox, sort_keys=True),
                            embedding_blob,
                            float(np.linalg.norm(embedding)),
                            int(time.time()),
                        ),
                    )
                    inserted_rows += 1

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
                "bbox": json.loads(bbox_json) if bbox_json else {},
                "embedding": arr.astype(np.float32),
                "embedding_norm": float(embedding_norm) if embedding_norm is not None else float(np.linalg.norm(arr)),
            }
        )

    return faces


def _to_similarity_percent(similarity: float) -> float:
    return round(max(0.0, min(100.0, float(similarity) * 100.0)), 2)


def _to_distance_percent(similarity: float) -> float:
    return round(max(0.0, min(100.0, (1.0 - float(similarity)) * 100.0)), 2)


def match_source_to_db(
    source_image: Path,
    db_path: Path,
    top_n: int = 5,
    model_name: str = "ArcFace",
    detector_backend: str = "retinaface",
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
        distance = float(np.linalg.norm(source_embedding - candidate_embedding))
        similarity = 1.0 / (1.0 + distance)
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

    scored_matches.sort(key=lambda item: (float(item["_raw_distance"]), -float(item["_raw_similarity"])))
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

    matches.sort(key=lambda item: (float(item["_raw_distance"]), -float(item["_raw_similarity"])))
    for item in matches:
        item.pop("_raw_distance", None)
        item.pop("_raw_similarity", None)
    return matches[:top_n]


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Match a source portrait against indexed candidate faces or build a face index from a target directory.",
)
def main() -> None:
    """CLI entry point for indexing and matching portraits."""
    return


@main.command("scan")
@click.argument("target_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
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
    target_dir: Path,
    db_path: Path,
    model_name: str,
    detector_backend: str,
    overwrite: bool,
) -> None:
    """Scan a directory of portraits and create a SQLite face index."""
    try:
        inserted = index_target_directory(
            target_dir=target_dir,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            overwrite=overwrite,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Indexed {inserted} faces into {db_path}")


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
    json: bool,
) -> None:
    """Match a source portrait against faces already indexed in SQLite."""
    try:
        results = match_source_to_db(
            source_image=source_image,
            db_path=db_path,
            top_n=top_n,
            model_name=model_name,
            detector_backend=detector_backend,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if json:
        click.echo(json.dumps(results, indent=2))
        return

    for idx, result in enumerate(results, start=1):
        click.echo(f"{idx}. {result['path']} | distance={result['distance']}% | similarity={result['similarity']}%")


if __name__ == "__main__":
    main()
