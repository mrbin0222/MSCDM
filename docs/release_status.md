# Release status and validation scope

The MSCDM repository was publicly released on 15 October 2026. It contains the method implementation, configurations, fixed sample and condition metadata, evaluation tools, documentation and constructed examples. Publication of the repository does not establish complete end-to-end reproduction of the paper from this extracted package.

## Distribution scope

The software is released under the MIT license with the MSCDM contributors copyright notice and applicable third-party notices. Study-trained weights, third-party pretrained models, original images and annotations are not distributed. Users obtain datasets and external model assets from their providers and supply compatible checkpoints for inference. These exclusions are the release scope, not pending uploads. Exact reproduction of archived models also depends on the corresponding initialization and checkpoints.

## Completed validation

Checks were performed in an independent directory on the research server using its existing generation and Cellpose environments:

- All 37 functional command-line entry points loaded; each environment passed 17 data-free tests.
- The compact catalog retained the required contents and record order of the expanded manifests. The three condition lists and label vocabulary were unchanged, and runtime manifest materialization passed.
- Installation into an isolated target directory without downloading or changing dependencies, followed by installed-package imports, passed.
- The paper foreground checkpoint loaded strictly. Extracted and original foreground tensor paths agreed bitwise on constructed RPE, 3T3 and same-class inputs.
- Constructed-fixture checks covered CTC request preparation, planning without a historical generated cache and instance-matching metrics.
- After publication, a fresh clone from GitHub matched all 170 candidate files by SHA-256.

These checks establish package and interface consistency within the tested environments. They do not constitute a fresh dependency installation, full-resolution image-inference equivalence or downstream reproduction.

## Reproduction checks not yet completed

- **Complete provider conversion:** the converters for all four provider families are included and their command-line interfaces loaded, but conversion of the complete original datasets has not been rerun from this package.
- **Fresh environments:** clean Linux generation/evaluation and Cellpose 4.2 dependency installations have not been validated.
- **Full-resolution foreground inference:** the public adapter has not been compared against the original image-inference call using identical weights, masks, backgrounds and noise. The small constructed-tensor comparison above does not replace this check.
- **End-to-end segmentation:** full-resolution preparation, generation and downstream execution with real CTC inputs have not been validated from the extracted package. The preparation interface accepts the converted CTC manifest and can plan without a historical generated cache.
- **Turnkey evaluation:** evaluator code, fixed labels and vocabulary are included, and runtime manifests can be materialized locally. Reproducing the final evaluation also requires the corresponding weight assets and selection protocol; a complete run from the released package remains unverified.

## Further maintenance

- Validate per-entry command-line configurations for all recorded paper settings; JSON metadata and CLI options are currently provided separately, with entry points mapped in [experiments.md](experiments.md).
- Remove unused optional branches from shared training utilities after behavioral checks. Checkpoint/schema identifiers and some legacy settings remain for compatibility.
- Retain third-party notices and dataset access, attribution and use conditions. Any future data examples require an explicit redistribution basis; the current examples are constructed and contain no microscopy data.
