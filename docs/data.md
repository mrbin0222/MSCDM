# Data and fixed conditions

Obtain original data from their providers:

- BBBC021 v1: https://bbbc.broadinstitute.org/BBBC021
- IDR0033 screenA: https://doi.org/10.17867/10000105
- TissueNet v1.1: https://deepcell.readthedocs.io/en/latest/data-gallery/tissuenet.html
- DynamicNuclearNet segmentation v1.0: https://deepcell.readthedocs.io/en/latest/data-gallery/dynamicnuclearnet.html
- Cell Tracking Challenge Fluo-N2DL-HeLa training images and ST annotations: https://celltrackingchallenge.net/2d-datasets/

Four public datasets form five image configurations; TissueNet whole-cell and nuclear views are not separate data sources. The CTC data are used only in the downstream experiment. Cite both CTC papers: https://doi.org/10.1038/nmeth.4473 and https://doi.org/10.1038/s41592-023-01879-y. CTC pixels and annotations are not redistributed here.

## Included metadata

`configs/manifests/samples.jsonl` is the single fixed sample index: 37,500 IDs with unchanged train/validation/test membership (25,000/2,500/10,000). `catalog.json` stores shared source and parent references, measured attribute categories, fixed labels and verbatim text dictionaries. The sample index refers to these tables rather than repeating long strings in several files. See [manifest format](manifest_format.md).

Generation, evaluation and foreground-adaptation texts are distinct stored inputs. Their original values and profile-specific record order are preserved; the exporter does not synthesize text or run a language model. Expanded reconstruction, evaluator and foreground manifests are generated locally:

```bash
python scripts/data/materialize_manifests.py --output-dir data/processed
```

This produces the runtime training manifest, reconstruction metadata, full mask references, evaluator labels/vocabulary and foreground training input. It creates metadata only, not images or masks. Existing files with different contents are preserved with an error.

`validation_diagnostics_50.jsonl`: 50 cross-class validation conditions shared by the adaptation comparisons; target classes are imbalanced. `fixed_geometry_test_180.jsonl`: 36 test anchors, each requesting the other five classes, 16 generations per condition. `auxiliary_test_200.jsonl`: a separate test-text condition set, using geometries predicted from generated images in the auxiliary protocol. Do not combine their denominators. Condition repeats are not training seeds.

## Reconstruction

The inherited crop entry expects prepared parent RGB images, integer instance maps and source manifests in the source-specific layout documented by its CLI; it does not directly read every provider's archive format. The original BBBC021, IDR0033 and DeepCell parent-data converters are now included; the CTC archive reader is included as well. Their required inputs differ: BBBC021 needs images and measurement/instance-map files, IDR0033 needs the screen/outline layout, DeepCell uses released NPZ archives, and CTC reads the provider ZIP. These were not rerun over complete raw datasets in this preparation.

```bash
python scripts/data/crop_instances.py --help
python scripts/data/isolate_instances.py --help
python scripts/data/attach_released_text.py --input data/rebuilt/manifest.jsonl --output data/rebuilt/manifest_hybrid.jsonl
```

Use the released sample IDs/splits to select instances, not a new random split. `crop_instances` performs candidate cropping; `isolate_instances` applies background reconstruction to the selected manifest. The code exposes thresholds explicitly: minimum area 32 pixels and side length 6, boundary rejection, crop padding 0.30, and background exclusion/dark-pool settings. Parent contribution caps and source-balanced selection must match the paper manifest, not unmodified CLI defaults. Do not regenerate text via a current language model: the released text is the actual fixed input. Text describes operational entity/geometry, within-source class/treatment and background/acquisition fields.

The isolated background uses same-parent pixels outside annotated instances, with exclusion dilation and dark-pool sampling. Preserve instance IDs, normalization and random seeds to reconstruct pixels. Upstream parent preprocessing also matters; a matching text or ID alone does not establish pixel equivalence.

## Included provider conversion entry points

```bash
python scripts/data/prepare_bbbc021.py --help
python scripts/data/prepare_idr0033.py --help
python scripts/data/prepare_deepcell.py --help
python scripts/data/prepare_ctc_archives.py --raw_dir /path/to/CTC --datasets Fluo-N2DL-HeLa --annotation_source st --out_dir data/ctc_segmentation_st_v1
```

These source converters were retained from the project's upstream preprocessing implementation. The small metadata/bounding-box helpers they need were extracted, without including the unrelated upstream model.

The generated `data/processed/control_metrics_v2/evaluator_manifest.jsonl` and `label_vocab.json` provide fixed evaluator labels and vocabulary. `data/processed/foreground_samples.jsonl` provides 10,000 training and 1,000 validation foreground records; `mask_manifest.jsonl` covers all sample mask references for evaluation. No learned predictions or image/annotation pixels are included. Image and mask references become usable only after the separate data preprocessing has created the referenced files.

## Distribution policy

Original images and annotations must be downloaded from the providers above by each user. They are not pending repository uploads. Study-trained weights are also excluded from this code-only release. Software licensing does not replace any dataset access, attribution or use condition.
