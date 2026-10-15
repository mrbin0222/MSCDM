# Model scope

The text pathway consists of three latent scales and separate VAE/U-Net components with structured text routing. Hierarchical and joint routing both use the same three-scale architecture. The foreground pathway uses a high-resolution VAE latent and a nine-channel denoiser input: noisy latent (4), background latent (4), and support mask (1). Target-specific FiLM acts within the U-Net; request gating enables dedicated responses when appropriate.

Distribution-teacher supervision, parameter interpolation and sparse modulation training have separate roles. The classifier and teacher are not needed for final inference. The release preserves checkpoint parameter names and schema identifiers where required for existing checkpoint compatibility. Internal compatibility keys must not be interpreted as separate public methods.

Inputs and comparison budgets differ across tasks. Lower FID does not establish universal reference fidelity or attribute control. The evaluation includes class failures and disagreements between evaluation representations. Downstream gains concern the reported HeLa setting with preserved real spatial context. Operational labels are not independent biological identity/function ground truth.

No model weights are included or promised by this package. Checkpoint filenames and resource roles are input conventions for user-supplied assets, not pending uploads or download offers. Small CPU tensor paths agree bitwise with the original implementation. Full-resolution end-to-end and GPU numerical equivalence remain unverified after extraction.
