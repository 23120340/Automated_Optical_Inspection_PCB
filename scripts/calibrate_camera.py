"""Create a camera-calibration JSON profile from chessboard photographs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aoi_pipeline import (  # noqa: E402
    CalibrationProfileError,
    calibrate_from_chessboards,
    load_image,
    save_calibration_profile,
)


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a pinhole camera profile from at least 10 chessboard images. "
            "Columns/rows are INNER corner counts."
        )
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image files, directories, or glob patterns containing calibration images.",
    )
    parser.add_argument("--columns", type=int, default=9, help="Inner corners per row (default: 9).")
    parser.add_argument("--rows", type=int, default=6, help="Inner corners per column (default: 6).")
    parser.add_argument(
        "--square-size",
        type=float,
        default=1.0,
        help="Physical chessboard square size in any consistent unit (default: 1.0).",
    )
    parser.add_argument("--camera-id", default=None, help="Optional stable camera/serial identifier.")
    parser.add_argument("--lens-id", default=None, help="Optional lens/focal-length identifier.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("camera_calibration.json"),
        help="Destination JSON profile (default: camera_calibration.json).",
    )
    return parser


def resolve_image_paths(inputs: list[str]) -> list[Path]:
    resolved: set[Path] = set()
    for raw in inputs:
        candidate = Path(raw).expanduser()
        matches: list[Path]
        if candidate.is_dir():
            matches = [path for path in candidate.rglob("*") if path.is_file()]
        elif candidate.is_file():
            matches = [candidate]
        else:
            matches = [Path(path) for path in glob.glob(raw, recursive=True)]
        for path in matches:
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
                resolved.add(path.resolve())
    return sorted(resolved, key=lambda path: str(path).casefold())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = resolve_image_paths(args.images)
    if not paths:
        parser.error("No supported calibration images were found.")
    try:
        run = calibrate_from_chessboards(
            [(str(path), load_image(path)) for path in paths],
            pattern_size=(args.columns, args.rows),
            square_size=args.square_size,
            camera_id=args.camera_id,
            lens_id=args.lens_id,
        )
        output = save_calibration_profile(run.profile, args.output)
    except (CalibrationProfileError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Profile: {output}")
    print(f"Accepted images: {len(run.accepted_images)}")
    print(f"Rejected images: {len(run.rejected_images)}")
    print(f"RMS reprojection error: {run.profile.rms_reprojection_error:.6f}")
    print(f"Mean reprojection error: {run.profile.mean_reprojection_error:.6f}")
    if run.rejected_images:
        print("Rejected:")
        for name in run.rejected_images:
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
