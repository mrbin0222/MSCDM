# Paper workflows and commands

Run from the repository root. `--help` exposes the real parser of each extracted implementation; required weights and manifests must exist before execution. The recorded JSON files preserve settings, but they are **not** automatically consumed as a universal training config. Do not assume current parser defaults equal the paper settings.

| Scientific operation | Entry point |
|---|---|
| train text | `python scripts/generation/train_text.py --help` |
| infer text | `python scripts/generation/infer_text.py --help` |
| generate text dataset | `python scripts/generation/generate_text_dataset.py --help` |
| evaluate images | `python scripts/evaluation/evaluate_images.py --help` |
| crop instances | `python scripts/data/crop_instances.py --help` |
| isolate instances | `python scripts/data/isolate_instances.py --help` |
| attach text conditions | `python scripts/data/attach_text_conditions.py --help` |
| prepare foreground manifest | `python scripts/data/prepare_foreground_manifest.py --help` |
| cache foreground latents | `python scripts/data/cache_foreground_latents.py --help` |
| train foreground | `python scripts/generation/train_foreground.py --help` |
| train distribution teacher | `python scripts/adaptation/train_distribution_teacher.py --help` |
| interpolate backbone | `python scripts/adaptation/interpolate_backbone.py --help` |
| train rpe modulation | `python scripts/adaptation/train_rpe_modulation.py --help` |
| train 3t3 modulation | `python scripts/adaptation/train_3t3_modulation.py --help` |
| resynthesize foreground | `python scripts/generation/resynthesize_foreground.py --help` |
| edit generated geometry | `python scripts/generation/edit_generated_geometry.py --help` |
| prepare single-instance segmentation inputs | `python scripts/segmentation/prepare_single_instances.py --help` |
| segmentation prepare | `python scripts/segmentation/segmentation_prepare.py --help` |
| segmentation run | `python scripts/segmentation/segmentation_run.py --help` |
| segmentation metrics | `python scripts/segmentation/segmentation_metrics.py --help` |
| segmentation native predict | `python scripts/segmentation/segmentation_native_predict.py --help` |
| segmentation formal pool | `python scripts/segmentation/segmentation_formal_pool.py --help` |
| evaluate condition control | `python scripts/evaluation/evaluate_condition_control.py --help` |
| train attribute evaluators | `python scripts/evaluation/train_attribute_evaluators.py --help` |
| finetune attribute evaluators | `python scripts/evaluation/finetune_attribute_evaluators.py --help` |
| train foreground evaluator | `python scripts/evaluation/train_foreground_evaluator.py --help` |

## Training dependency order

1. Obtain provider inputs and reconstruct the fixed parent/instance dataset. Run `python scripts/data/materialize_manifests.py --output-dir data/processed` for runtime metadata, or use `attach_released_text.py` to attach the actual released text to a reconstructed manifest.
2. Train the text-conditioned model with the recorded model/training settings and the required VAE initialization. Train-only latent calibration is preserved in recorded metadata.
3. Prepare the foreground vocabulary/manifest, encode the frozen high-resolution VAE latent cache, and train the shared foreground model.
4. Train the adapted teacher classifier and construct the training-only six-dimensional standardized logit distribution bank (`scripts/adaptation/train_teacher_classifier.py`, `build_teacher_distribution.py`); use it to train the distribution teacher.
5. Interpolate the foreground and teacher parameters at equal weights. This creates one backbone, not an inference ensemble.
6. Train RPE modulation for the recorded two six-epoch blocks, retaining the continuation/optimizer semantics; train the 3T3 target response with the recorded six-epoch setting. Do not replace continuation with an unverified fresh twelve-epoch run and call it exact reproduction.
7. Evaluate with separate frozen attribute/foreground evaluators. Classifier hits are evaluation endpoints, not biological transformations.

## Direct foreground inference

```bash
python scripts/generation/edit_foreground.py \
  --checkpoint checkpoints/dual_target_modulation.pt \
  --vae-checkpoint checkpoints/text_generation.pt \
  --background data/example/background.png --mask data/example/mask.png \
  --anchor-class hela --target-class rpe --steps 25 --seed 42 \
  --output outputs/example.png
```

Background is RGB uint8, 512×512, with foreground-support pixels set to zero; mask is binary 512×512. The loader checks VAE calibration. This new public adapter follows the existing sampling call and requires a checkpoint-backed equivalence test before release. For same-class two-source resynthesis use `resynthesize_foreground` and its manifest interface.

## Segmentation

Use the full-scene preparation/assembly path (`segmentation_formal_pool`) and Cellpose-SAM runner. The earlier union-mask generation experiment is excluded. Both transfer directions use 92 parent frames; three fine-tuning arms use 2,000 updates and effective batch 8. Pretrained inference is not fine-tuned. Mixed batches contain four real and four synthetic examples. Sampling/assembly preserves real backgrounds, layouts and integer ST annotations; all positive instances must be rendered.

The request preparation entry now reads the converted CTC manifest directly. Full-scene planning takes the resulting single-instance preparation and does not require a historical corrected-generation cache. This CPU input chain was tested using constructed fixtures; complete image generation and segmentation training remain untested from the extracted package. Segmentation metrics implement instance matching TP/(TP+FP+FN) over the configured IoU thresholds, not COCO AP. `paired_summary.py` pairs training seeds and first averages directions equally; its SD is not a confidence interval.

## Segmentation preparation without historical result folders

From the repository root, place converted CTC data under `data/ctc_segmentation_st_v1`, then:

```bash
python scripts/segmentation/segmentation_prepare.py --ctc-data-dir data/ctc_segmentation_st_v1 --config configs/segmentation_protocol.json --out-dir outputs/ctc_requests
python scripts/segmentation/prepare_single_instances.py --source-package outputs/ctc_requests --config configs/segmentation_protocol.json --out-dir outputs/ctc_instances
python scripts/segmentation/segmentation_formal_pool.py plan --source outputs/ctc_instances --package outputs/ctc_pool --parent-root . --frames-per-direction 92 --crops-per-frame 4
```

`--corrected-batch` is optional cache reuse and should be omitted for a fresh reconstruction. Model resources remain separate: provide `checkpoints/dual_target_modulation.pt`, `checkpoints/text_generation.pt` and the Cellpose initialization specified by the configuration. `--parent-root` names the directory containing `data/ctc_segmentation_st_v1`; it does not import a sibling project's code.

## Fixed manifest interfaces

Expanded evaluator/reconstruction/foreground files are no longer distributed as redundant source tables. Generate them locally using `scripts/data/materialize_manifests.py`; see [manifest format](manifest_format.md) for the output paths. For foreground preparation:

```bash
python scripts/data/prepare_foreground_manifest.py --training_manifest data/processed/foreground_samples.jsonl --evaluator_manifest data/processed/control_metrics_v2/evaluator_manifest.jsonl --output_manifest data/processed/foreground_train_val.jsonl --audit_json data/processed/foreground_vocabulary.json
```

The foreground evaluator uses `data/processed/mask_manifest.jsonl` to resolve masks for all fixed splits. The foreground training table contains training/validation samples only.
