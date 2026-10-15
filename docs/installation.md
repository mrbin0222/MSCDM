# Environments

The inspected generation environment uses Python 3.10 and PyTorch 2.11.0+cu130. A separate Cellpose environment was inspected for its actual versions. The observed requirements files intentionally distinguish observed installations from tested installation instructions.

Install the core package with `pip install -e .`; use `pip install -e '.[preprocessing]'` for provider conversion and the complete data-free test suite. This adds scipy, scikit-image and tifffile. Evaluation and segmentation have separate extras as shown in the README.

Core dependencies: torch, numpy, Pillow, transformers, huggingface-hub, safetensors, tqdm. Evaluation additionally uses torchvision, torchmetrics, torch-fidelity, scipy and scikit-learn; some image tools use scikit-image and tifffile. Segmentation uses the Cellpose 4.2 API and its dependencies in a separate environment. Install platform-specific torch/torchvision together; do not use the CUDA 13 wheel on macOS.

Model assets are not bundled: CLIP ViT-B/32, CLIP ViT-L/14, DINOv2-L/14, Inception FID and Cellpose-SAM, plus study-trained checkpoints/evaluators. `configs/resources.json` is a path convention, not a download manifest. Do not substitute arbitrary weights while describing the run as the paper protocol.

No training, scoring or segmentation inference was run when preparing this package. The data-free test results are documented separately from model execution.

## Server validation on 2026-10-15

The extracted package was checked in an independent directory on the research server using the existing generation and Cellpose environments. After the completeness update, all 37 functional CLIs loaded; each environment passed seventeen tests, including compact-catalog equivalence and materialization checks. A no-dependency/no-index install into an isolated target directory succeeded, and both the installed text-model API and installed segmentation/Cellpose modules imported. This is validation against existing environments, not a fresh dependency installation.

An eager package import initially forced Transformers into the segmentation environment. Lazy loading of the text-generation model removed that unrelated dependency; no original research environment was modified. The final foreground checkpoint loaded strictly with 23,040 RPE and 3,840 3T3 modulation parameters. Extracted and original foreground tensor paths agreed bitwise for RPE, 3T3 and same-class requests under a small constructed input. This does not substitute for full-resolution checkpoint-backed image inference.

The code-only distribution policy excludes both study-trained weights and third-party model downloads. Required assets are not an unfinished upload task. Obtain original datasets and external model assets from their providers; inference requires compatible user-supplied checkpoints.
