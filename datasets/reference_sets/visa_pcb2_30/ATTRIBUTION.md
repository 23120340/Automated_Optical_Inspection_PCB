# Amazon VisA attribution

The images in this reference set are derived from the **Visual Anomaly
(VisA)** dataset by Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang,
and Onkar Dabeer (ECCV 2022).

- Official project: https://github.com/amazon-science/spot-diff
- Official AWS Open Data record: https://registry.opendata.aws/visa/
- Official archive: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
- Dataset license: CC BY 4.0
  https://creativecommons.org/licenses/by/4.0/
- Paper: https://arxiv.org/abs/2207.14315

This repository downloads a deterministic 30-image subset of `pcb2.train`
through the `BrachioLab/visa` Hugging Face convenience mirror. The downloaded
JPEG bytes are kept unchanged, and their SHA-256 digests are recorded in
`manifest.json`. The mirror access path is a redistribution/conversion layer;
the dataset authors and the official Amazon archive remain the upstream source.

Changes made here:

- selection of 30 evenly spaced normal-training rows;
- deterministic local filenames;
- a candidate Golden/reference selected from the real downloaded frames;
- draft detector-derived component consensus and pick-and-place files.

The draft annotations, Golden selection, and PnP files are project-generated
and are **not** annotations supplied or endorsed by the VisA authors.

