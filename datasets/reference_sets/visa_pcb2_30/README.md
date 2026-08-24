# VisA PCB2 — 30-image review set

This directory is a bootstrap set for AOI phases 1–5. All 30 images belong to
the VisA `pcb2` category and are labelled *normal* by the upstream anomaly
dataset. In this project, `pcb2` is treated as an SKU proxy for the bottom side
of an HC-SR04-style PCB.

Important limits:

- VisA does not publish physical board-instance or manufacturing-lot IDs.
- `normal` is an image-level anomaly label, not component/pad/solder ground truth.
- The images are `dataset_preprocessed` / source-as-received, not sensor RAW.
- Do not use this set alone to claim production solder-defect accuracy.

Generated layout:

```text
images/                  exact downloaded JPEG bytes
manifest.json            row indices, hashes, dimensions and provenance
reference_selection.json medoid selection scores and rejected frames
labels/                   place reviewed labels here
```

The selected Golden recipe and PnP draft live under
`golden_recipes/visa_pcb2/bottom/`. They remain `NEEDS_REVIEW` and
`production_eligible=false` until a human verifies the master image, board
dimensions, RefDes, footprint, rotation, anchors and calibration.

See [ATTRIBUTION.md](ATTRIBUTION.md) before redistributing the images.
