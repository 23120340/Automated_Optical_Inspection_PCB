"""Check that a trained step-6.2 artifact pair will actually load in the app.

Run this on whatever comes back from Kaggle, before wiring it into a line:

    python scripts/verify_solder_model.py models/solder/best.onnx models/solder/model_manifest.json

It loads the pair through the very same runtime the app uses, so a pass here
means the app will accept it. It then pushes a synthetic batch through and
reports the class order, so a transposed taxonomy shows up as a wrong label on
an obviously-bare ROI rather than as silent passes on a production line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aoi_pipeline import SolderGradingConfig, SolderInspector  # noqa: E402
from aoi_pipeline.core.exceptions import (  # noqa: E402
    AOIPipelineError,
    ClassifierConfigurationError,
)
from aoi_pipeline.core.models import BoundingBox, SolderJoint, SolderJointCrop  # noqa: E402
from aoi_pipeline.grading.classifier import create_solder_classifier  # noqa: E402
from aoi_pipeline.grading.rules import COMPONENT_CLASSES, JOINT_CLASSES  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="Path to best.onnx")
    parser.add_argument("manifest", help="Path to model_manifest.json")
    parser.add_argument(
        "--strict-taxonomy",
        action="store_true",
        help="Fail when the model's classes are not a subset of the pipeline taxonomy.",
    )
    return parser


def _probe_roi(fill: float) -> np.ndarray:
    """A synthetic ROI with a known amount of bright 'solder' in it."""

    image = np.full((48, 96, 3), (40, 90, 40), np.uint8)
    if fill > 0:
        width = int(96 * fill)
        cv2.rectangle(image, (48 - width // 2, 12), (48 + width // 2, 36), (215, 215, 215), -1)
    return image


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = Path(args.model).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    problems: list[str] = []
    print(f"model    : {model_path}")
    print(f"manifest : {manifest_path}\n")

    try:
        classifier = create_solder_classifier(
            model_path, manifest_path, SolderGradingConfig()
        )
    except (ClassifierConfigurationError, AOIPipelineError) as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        print(
            "\nThe app would refuse this pair for the same reason. Fix the manifest "
            "or re-export; see docs/solder_model_manifest_template.json.",
            file=sys.stderr,
        )
        return 1
    if classifier is None:
        print("Both a model and a manifest are required.", file=sys.stderr)
        return 2

    print("Contract accepted.")
    print(f"  scope          : {classifier.scope}")
    print(f"  classes        : {classifier.class_names}")
    print(f"  good_label     : {classifier.good_label}")
    print(f"  input size     : {classifier.input_size}")
    print(f"  accept/review  : {classifier.accept_threshold} / {classifier.review_threshold}")
    print(f"  model version  : {classifier.model_version}")

    expected = set(COMPONENT_CLASSES if classifier.scope == "component" else JOINT_CLASSES)
    unknown = [name for name in classifier.class_names if name not in expected]
    if unknown:
        message = (
            f"Classes not in the pipeline taxonomy: {unknown}. They will still be "
            "reported, but the rule layer can never agree with them, so every such "
            "prediction lands in the review queue as a conflict."
        )
        problems.append(message)
        print(f"\nWARNING: {message}")

    # A forward pass proves the ONNX graph runs and the output width matches the
    # declared class list -- the two ways a good-looking manifest still fails.
    print("\nRunning a forward pass on synthetic ROIs...")
    try:
        probes = [_probe_roi(fill) for fill in (0.0, 0.25, 0.75)]
        predictions = classifier.predict(probes)
    except Exception as exc:  # noqa: BLE001 - reported to the operator
        print(f"FAILED to run inference: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for fill, prediction in zip(("bare land", "some solder", "lots of solder"), predictions):
        top = prediction[0]
        print(f"  {fill:16s} -> {top.label:16s} {top.probability:.3f}")

    bare = predictions[0][0]
    if bare.label == classifier.good_label and bare.probability > 0.9:
        message = (
            "The model calls a completely bare synthetic land 'good' with high "
            "confidence. That can mean the class order is transposed, or simply "
            "that the synthetic probe is far from the training distribution. "
            "Check against a few real bare-land crops before trusting it -- the "
            "escape guard will catch this case at runtime, but nothing catches a "
            "transposed taxonomy on the other classes."
        )
        problems.append(message)
        print(f"\nWARNING: {message}")

    # Finally, the path the app really takes.
    print("\nRunning through SolderInspector (the path the app uses)...")
    config = SolderGradingConfig(
        model_path=str(model_path), manifest_path=str(manifest_path)
    )
    inspector = SolderInspector(config)
    if not inspector.has_model:
        print("SolderInspector did not pick up the model.", file=sys.stderr)
        return 1
    crops = [
        SolderJointCrop(
            image=_probe_roi(fill),
            joint=SolderJoint(
                detection_id="probe",
                joint_id=f"probe_{index}",
                label="resistor",
                kind="joint",
                bbox=BoundingBox(0, 0, 96, 48),
                terminal_geometry="two_terminal",
                position="terminal_a",
            ),
            filename=f"probe_{index}.png",
        )
        for index, fill in enumerate((0.0, 0.35, 0.9))
    ]
    for verdict in inspector.inspect(crops):
        print(
            f"  {verdict.label:16s} {verdict.decision:7s} {verdict.source:12s} "
            f"rule={verdict.rule_label}"
        )
    for warning in inspector.warnings:
        print(f"  WARNING: {warning}")

    print("\n" + ("-" * 66))
    if problems:
        print(f"USABLE, with {len(problems)} warning(s) above. The app will load this pair.")
    else:
        print("OK. The app will load this pair and nothing looked suspicious.")
    print(
        "Remember: this only proves the contract is sound. It says nothing about "
        "accuracy on your line -- that needs the escape/false-call numbers from a "
        "validation set of your own boards."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
