# Per-board Golden/PnP bootstrap

The 30 source images are different layouts. Each folder under `references/` is independent; do not build a cross-board consensus.

- `golden_candidate.json`: source reference identity and review gates.
- `pnp_pixels_NEEDS_REVIEW.csv/json`: pixel-native pseudo-labels only.
- `pnp_preview.jpg`: green boxes are upstream IC annotations; orange boxes are detector proposals.
- `reference_index.csv`: image identity, focus score and proposal counts.
- `contact_sheet.jpg`: diversity/quality review for all 30 boards.

No file here contains verified millimetres, manufacturing RefDes, footprints, board side or production acceptance. Follow `../NHUNG_VIEC_BAN_CAN_LAM.md` before promoting any artifact.
