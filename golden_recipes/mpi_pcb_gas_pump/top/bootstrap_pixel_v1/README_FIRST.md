# Draft Golden/PnP — review before use

Status: **NEEDS_REVIEW**. This bundle is not production-ready.

- Golden is one real source frame: `mpi_pcb_train_good_0893.jpg`; `golden.png` is its
  lossless decoded copy, not a median/composite image.
- `consensus_components.json/csv` is the complete generated detector audit in
  `golden_board_pixels`; it is not human-verified component ground truth or PnP.
- `pnp_pixels_NEEDS_REVIEW.csv` is a non-authoritative proposal queue. It contains
  exactly the consensus sites observed in the selected Golden frame, while its
  centers and boxes remain multi-frame medians. Synthetic designators, rotation,
  footprint and component identity still require human review.
- `overlay_pnp_NEEDS_REVIEW.png` draws only that selected-Golden-anchored subset.
- No physical board dimensions or fiducial registration were supplied. The
  millimetre placement CSV, metrology registration and `recipe.json` are
  intentionally absent rather than populated with guessed values.
- Measure the real board and register fiducials/CAD before exporting PnP in mm
  or creating an inspection recipe. Until then, use only the pixel-native audit
  and review artifacts.

Green boxes in `overlay_consensus.png` passed the configured multi-frame
support/purity gate; orange boxes did not. Inclusion in
`pnp_pixels_NEEDS_REVIEW.csv` is instead based on presence in the selected
Golden, so either gate colour can occur. Every PnP review row still requires
human verification.
