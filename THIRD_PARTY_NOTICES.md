# Third-party notices

## What is distributed

This candidate contains project source code, configurations, documentation,
metadata-only manifests, constructed examples and tests. It does not bundle
third-party package source distributions, model weights, dataset images or
annotation archives. The repository's MIT license does not change
third-party terms or relicense dataset content.

## External software and model sources

| Component | Observed source license | Official source |
|---|---|---|
| CLIP code | MIT | https://github.com/openai/CLIP/blob/main/LICENSE |
| DINOv2 code | Apache-2.0 | https://github.com/facebookresearch/dinov2/blob/main/LICENSE |
| Cellpose 4.2.1.1 code | Three-condition BSD-style license; copyright HHMI | https://github.com/MouseLand/cellpose/blob/v4.2.1.1/LICENSE |

These observations concern the cited source files, checked 2026-10-15. They
are not a blanket licensing conclusion for every separately downloaded
checkpoint. Follow the terms accompanying the exact model asset obtained.

The software also depends on PyTorch, torchvision, Transformers, NumPy,
Pillow, tqdm, SciPy, scikit-image, scikit-learn, torchmetrics, torch-fidelity,
safetensors, huggingface-hub and tifffile. Install them separately with their
own license files and notices. No dependency is relicensed by this project.

## Reused data

Provider links and dataset identities are in docs/data.md. Obtain raw images
and annotations directly from those providers under their access/use terms.
For the Cell Tracking Challenge data, retain source attribution and cite
both https://doi.org/10.1038/nmeth.4473 and
https://doi.org/10.1038/s41592-023-01879-y. This repository does not mirror CTC
images or annotations. Dataset-derived metadata retain applicable source
conditions; the software MIT license is not a license to the
underlying datasets.

## Attribution

Retain any third-party copyright/license notices attached to code when
copying or adapting it. The project source scan and file-origin mapping do
not establish exhaustive line-by-line authorship. A dependency attribution
list does not itself grant rights to another party's code or data.
