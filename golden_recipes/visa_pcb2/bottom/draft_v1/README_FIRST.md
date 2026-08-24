# Draft Golden/PnP — review before use

Status: **NEEDS_REVIEW**. This bundle is not production-ready.

- Golden is one real source frame: `row_000589.jpg`; `golden.png` is its
  lossless decoded copy, not a median/composite image.
- `consensus_components.json/csv` is the authoritative detector observation in
  `golden_board_pixels`.
- `placement_draft_NEEDS_REVIEW.csv` uses a provisional 45 x
  20 mm outline. Its AUTO designators are not OCR RefDes, and
  rotation/footprint are intentionally blank.
- `registration_draft_NEEDS_REVIEW.json` uses visual top-left as origin, X to
  visual right and Y down. Confirm mirroring for the photographed board side.
- Recipe anchors are demo grid patches and metrology is unverified, therefore
  `recipe.json` must remain `production_eligible=false`.

Green boxes in `overlay_consensus.png` passed the configured multi-frame
support/purity gate. Orange boxes remain in the pixel audit but were excluded
from the default PnP rows.
