# MSCDM

Research implementation for **Structured conditioning and multiscale generation of cell microscopy images for segmentation data augmentation**.

The associated manuscript is currently **under review**.

MSCDM combines structured text conditioning with multiscale diffusion to generate cell microscopy images. Given contours and backgrounds, a shared foreground backbone uses request-gated internal modulation to control foreground appearance. The repository contains the model, preprocessing, training and inference workflows, evaluation tools, and downstream segmentation workflows.

## Organization

- `src/mscdms`: multiscale diffusion, U-Net, VAE, text conditioning, foreground generation and internal workflow modules.
- `scripts`: functional data, generation, adaptation, evaluation and segmentation entry points.
- `configs`: model settings, runtime protocols and a compact fixed-input catalog.
- `docs`: installation, data reconstruction, experiment commands, model scope and remaining dependencies.
- `tests`, `examples`: data-free checks and a constructed paired-statistics example.

## Install

Use Python 3.10 or newer in an isolated environment. Install a compatible PyTorch/torchvision pair for your operating system and CUDA version first, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[preprocessing]'
# Optional: pip install -e '.[evaluation]'
# Install segmentation in a separate environment: pip install -e '.[segmentation]'
python scripts/check_environment.py
python -m unittest discover -s tests -v
```

The `requirements-*-observed.txt` files record packages actually found in the research environments. They are not clean-install lockfiles; CUDA builds must match the target platform. See [installation](docs/installation.md).

## Start here

```bash
python scripts/data/materialize_manifests.py --output-dir data/processed
python scripts/generation/edit_foreground.py --help
python scripts/data/attach_released_text.py --help
python scripts/evaluation/paired_summary.py --csv examples/paired_scores.csv
```

The example contains constructed scores, not paper results. Model inference requires external weights and processed inputs. Provider converters for BBBC021, IDR0033, TissueNet/DynamicNuclearNet and CTC are included under `scripts/data`. Full command mapping and input contracts are in [experiments](docs/experiments.md), [data](docs/data.md) and [model card](docs/model_card.md).

## Reproduction scope

Text generation has no reference geometry. Foreground generation accepts explicit masks and backgrounds; its dedicated RPE/3T3 modulation is activated by reference and target class requests inside the U-Net. The teacher and classifier are training dependencies, not inference-time ensembles. Segmentation augmentation resynthesizes original-class appearances while preserving real backgrounds, layouts and ST labels.

The package preserves research implementations rather than claiming retraining is byte-identical after extraction. Several file-based workflow interfaces still need external resources and path configuration. See [remaining dependencies](docs/release_status.md) before attempting end-to-end reproduction.

## Citation and license

If you use MSCDM in your research, please cite this GitHub repository:

```bibtex
@misc{shen2026mscdm,
  author={Shen, Bin and Xu, Hang and Gu, Zhen and Zhou, Jiale and Yan, Bingyong and Wang, Huifeng},
  title={Structured conditioning and multiscale generation of cell microscopy images for segmentation data augmentation},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/mrbin0222/MSCDM},
  note={Associated manuscript under review}
}
```

The citation will be updated to the journal article after publication.

Software, configurations and documentation are licensed under [MIT](LICENSE). Third-party dependencies retain their respective licenses. Dataset sources and their applicable terms are described in [the data guide](docs/data.md). Please credit the MSCDM research team when using this work.
