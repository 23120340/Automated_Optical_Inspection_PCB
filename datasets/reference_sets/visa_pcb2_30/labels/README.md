# Reviewed labels go here

Keep the files in `../images/` unchanged: `manifest.json` binds their exact
downloaded bytes and SHA-256 hashes. Store human-reviewed annotations under
this directory, paired by image stem (`row_000000`, `row_000031`, ...).

Before using an annotation for training, record at least:

- annotation format/schema and class list;
- annotator/reviewer and review status;
- whether boxes describe components, pads/pins, solder joints or defects;
- the image coordinate space and any transform applied by the labeling tool.

Detector proposals and PnP AUTO designators are drafts, not ground truth. Do
not copy them here as approved labels without reviewing every object and box.
