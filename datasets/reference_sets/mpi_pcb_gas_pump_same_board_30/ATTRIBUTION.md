# Attribution — MPI-PCB Dataset

The images in this directory are selected byte-for-byte from the aligned,
unmodified (`train/good`) portion of the **MPI-PCB Dataset**.

- Dataset record: https://zenodo.org/records/8213098
- Dataset DOI: https://doi.org/10.5281/zenodo.8213098
- Source archive: `original_aligned.zip`
- Source archive MD5 declared by Zenodo: `c49e03b709b2cddfd6c6344017a1afea`
- Associated paper: https://doi.org/10.3390/s23031353
- Authors: Diulhio Candido de Oliveira, Bogdan Tomoyuki Nassu, and Marco
  Aurelio Wehrmeister
- License declared by Zenodo: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/)

The archive-level MD5 is recorded but cannot be verified by a selective Range
download. Each selected member is instead checked against its ZIP central-
directory CRC32 after decompression and receives a local SHA-256 in
`manifest.json`.

The upstream record describes these aligned images as repeated views of an
unmodified PCB from a gas pump. This supports same-layout enrollment, but the
public `good` label does not by itself approve a production Golden. Review all
frames and save the chosen canonical Golden as lossless PNG/TIFF.
