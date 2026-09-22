from __future__ import annotations

import json
from pathlib import Path

import click
from deepface import DeepFace

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _list_candidate_images(target_dir: Path) -> list[Path]:
    return sorted(
        path for path in target_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


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
                "distance": round(distance, 6),
                "similarity": round(similarity, 6),
            }
        )

    if not matches:
        raise ValueError(
            "No valid face matches were found. Check that the source and target images contain detectable faces."
        )

    matches.sort(key=lambda item: (float(item["distance"]), -float(item["similarity"])))
    return matches[:top_n]


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Compare a source portrait to a directory of portraits and return the closest matches.",
)
@click.argument("source_image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("target_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--top-n",
    default=5,
    show_default=True,
    type=click.IntRange(1),
    help="Return the N closest matches from the target directory.",
)
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
    help="Detector backend used to find the face in each image.",
)
@click.option(
    "--json",
    is_flag=True,
    default=False,
    help="Emit the results as JSON instead of a text table.",
)
def main(
    source_image: Path,
    target_dir: Path,
    top_n: int,
    model_name: str,
    detector_backend: str,
    json: bool,
) -> None:
    """CLI entry point for comparing a source portrait to a directory of target portraits."""
    try:
        results = find_top_matches(
            source_image=source_image,
            target_dir=target_dir,
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
        click.echo(f"{idx}. {result['path']} | distance={result['distance']} | similarity={result['similarity']}")


if __name__ == "__main__":
    main()
