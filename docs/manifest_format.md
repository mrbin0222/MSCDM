# Fixed-input catalog and generated manifests

## Repository files

| File | Role |
|---|---|
| `samples.jsonl` | One record per selected instance: ID, split, source/parent references, instance number, bounding box and dictionary references |
| `catalog.json` | Shared dictionaries of actual text, source descriptors, parents, attributes, fixed labels and foreground control labels; original order for each runtime profile |
| `label_vocab.json` | Fixed evaluation class names and IDs |
| `validation_diagnostics_50.jsonl` | The separate 50-condition validation diagnosis set |
| `fixed_geometry_test_180.jsonl` | 36 test anchors × five other target classes |
| `auxiliary_test_200.jsonl` | The separate 200-condition auxiliary test set |

The three condition files and the label vocabulary retain their original contents. Text dictionary entries are exact strings from the fixed manifests, not new templates. Generation, evaluation and foreground-adaptation text references are stored separately. Table indexes are local serialization references, not classes, biological identities or evaluation samples.

## Read one decoded sample

```python
from mscdms.manifests import ManifestCatalog
catalog = ManifestCatalog('configs/manifests')
sample = next(catalog.iter_profile('samples'))
print(sample['sample_id'], sample['split'], sample['subject'])
```

Profiles are `samples`, `reconstruction`, `evaluator_manifest` and `foreground_samples`. Each restores its original row order. The training profile contains the actual generative text; the evaluator profile restores its own text and canonical aliases. The foreground profile restores its own text, mask references, morphology and control labels. These inputs must not be substituted for one another.

## Generate local runtime files

Run from the repository root, using the environment for the relevant workflow:

```bash
python scripts/data/materialize_manifests.py --output-dir data/processed
```

| Generated file | Consumer |
|---|---|
| `manifest_hybrid.jsonl` | Text-generation training and dataset generation; actual text joined to necessary reconstruction metadata |
| `samples.jsonl` | Expanded fixed sample/text table |
| `reconstruction.jsonl` | Parent/instance correspondence and measured attributes |
| `control_metrics_v2/evaluator_manifest.jsonl` | Attribute/foreground evaluator and teacher classifier/distribution tools |
| `control_metrics_v2/label_vocab.json` | Evaluation vocabulary |
| `foreground_samples.jsonl` | Foreground manifest/vocabulary preparation (10,000 train + 1,000 validation) |
| `mask_manifest.jsonl` | Full fixed sample-to-mask references for foreground evaluation |
| `conditions/*.jsonl` | Copies of the three independent condition sets |

The public training/evaluation defaults point to `data/processed`. Images remain under its `images/` and masks under `target_masks/`, created separately from provider inputs. Metadata export does not download data, create images/masks, run generation or fit evaluators. The text-attachment entry reads the compact catalog directly; `--texts` also accepts an externally expanded manifest.

Repeated identical files may be reused; existing files with different contents trigger an error. Do not edit dictionary indexes by hand or regenerate splits. Semantic fingerprints describe the fixed expanded interfaces and include record order. Changing JSON whitespace does not alter these fingerprints.

## Deliberately excluded fields

The unused `morphology_mask_surrogate.after` and `.diagnostics` measurements are not stored in the fixed-input catalog. They are preprocessing diagnostics rather than model inputs. Mask references, the enabled flag, source ID, target morphology and required control labels are retained. Optional geometry preprocessing can calculate diagnostics when run; the runtime catalog does not reconstruct old diagnostic reports. Those diagnostics are not a dependency for the manuscript's foreground training/evaluation interface.

Generated manifests should remain in local ignored data/output directories. Do not commit the expanded copies alongside the catalog.

Expanded JSON whitespace and omitted diagnostic fields can change byte hashes relative to old files. Semantic fingerprints verify the necessary values and original order. External caches that bind to a specific manifest file hash should be built consistently from the generated interfaces rather than copied across different formats.
