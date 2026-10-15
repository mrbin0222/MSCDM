from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MSCDMConfig:
    """与文档表 4-3 对齐的主要超参数。"""

    # 图像与潜空间（表 4-1 / 4-3）
    image_sizes: tuple[int, int, int] = (128, 256, 512)
    latent_channels: int = 4
    # 各尺度潜空间空间分辨率 (H/8, W/8)
    latent_hw: tuple[tuple[int, int], ...] = ((16, 16), (32, 32), (64, 64))
    # 固定 VAE 潜变量的逐分支、逐通道 affine normalization。
    # z_diff = (z_vae - mean) / std；解码前执行逆变换。
    latent_normalization: bool = False
    latent_mean_low: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    latent_std_low: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    latent_mean_mid: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    latent_std_mid: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    latent_mean_high: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    latent_std_high: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)

    # 文本条件维度
    text_dim: int = 256
    max_text_len: int = 64
    # 训练服务器按整个 exp4 目录离线运行；禁止运行时访问 Hugging Face。
    text_pretrained_name: str = "models/clip-vit-base-patch32"
    text_local_files_only: bool = True
    freeze_clip_backbone: bool = True
    # hierarchical: low=subject, mid=state, high=background（v4 原始路径）
    # cumulative: low=subject, mid=subject+state, high=subject+state+background
    # semantic_multiscale: low=subject+background，mid/high=state+background；
    # 用于 observation scope 同时包含粗结构与高频成像语义的数据。
    # semantic_state_allscale: low=subject+state+background，mid/high 保持
    # state+background；用于 state 同时改变粗形态的数据。
    # semantic_state_seq_low: low token序列仍为subject+state+background，但
    # low pooled只保留subject+background；mid/high与allscale相同。
    # semantic_state_factor_residual_pool: sequence与allscale完全相同；
    # pooled保留原allscale均值，并叠加三因子各自的零初始化残差映射。
    # semantic_state_factor_partition_attention: route与allscale完全相同；
    # cross-attention额外计算各因子段内softmax并用零初始化标量门与原联合
    # attention插值，初始函数严格保持allscale，不改变pooled mean。
    # semantic_state_factor_isolated_combined: 同时启用上述分因子残差
    # pooled composition 与分因子token attention；两套新增参数均从零开始，
    # 初始函数仍严格等于allscale。
    # subject_state_low_nucleus: nucleus 的 low=subject+state，whole-cell 保持 subject
    # subject_state_low_nucleus_adapter: 冻结基座，仅 nucleus 低分支增加残差适配器
    # subject_state_low_nucleus_multiscale_adapter: nucleus 的 low/mid/high 均增加适配器
    # subject_state_low_nucleus_prompt_adapters: 冻结基座，在三路文本条件进入 U-Net 前
    # 使用 subject+state/state/background 的轻量门控 token adapter
    # hierarchical_nucleus_distill_prompt_adapters: 保持原 hierarchical token
    # 长度，只对 nucleus 使用带 subject/state/background 上下文的轻量 adapter；
    # whole-cell 路径严格恒等，用于跨 VAE 的图像空间教师蒸馏
    # hierarchical_nucleus_distill_output_adapters: 保持原 hierarchical 条件，
    # 只对 nucleus 的 low/mid/high 噪声预测增加时间与文本条件残差；
    # whole-cell 在预测端严格旁路
    # subject_state_low_nucleus_temporal_multiscale_adapter: 保持原 hierarchical
    # 条件主干，只对 nucleus 且 t<=阈值的低噪声后期启用三尺度残差 adapter
    # hierarchical_factor_prompt_adapters: 冻结 hierarchical 基座，只训练
    # subject/state/background 三路独立低秩 token adapter
    # hierarchical_factor_delta_adapters: 普通直接生成保持 hierarchical 原路径；
    # 仅在显式反事实编辑时，对目标因子注入 target-anchor 条件差分 adapter
    # hierarchical_factor_attention_delta_adapters: 普通生成继续旁路；
    # 显式编辑时只在目标尺度 U-Net 的 cross-attention K/V 中注入低秩差分
    # hierarchical_rgb_factor_output_adapters: 冻结 hierarchical 基座，
    # morphology 与 observation 各自通过三尺度零初始化残差只承担伪彩 RGB
    # latent 的生成责任；不输出、不重建也不监督任何 raw channel
    # hierarchical_morphology_latent_bank: subject 文本仅保留 entity；
    # round/oval/elongated 由共享的低分辨率多通道 latent bank 独占表示
    # hierarchical_morphology_feature_affine: subject 文本仅保留 entity；
    # 三类形态通过 low U-Net 各层的零初始化 feature affine decoder 注入
    # hierarchical_morphology_spatial_affine: subject 文本仅保留 entity；
    # 文本选择的目标形态模板在 low U-Net 各层解码为空间 feature affine
    conditioning_mode: str = "hierarchical"
    # semantic_state_specific_residual 使用的确定性两层 state 专用残差秩。
    state_specific_residual_rank: int = 64
    factor_attention_delta_rank: int = 4
    nucleus_temporal_adapter_max_timestep: int = 249

    # U-Net
    base_channels: tuple[int, int, int] = (192, 224, 256)
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (32, 16, 8)
    num_heads: int = 8
    unet_context_dim: int = 256
    dropout: float = 0.0
    use_reversible: bool = True
    reversible_save_memory: bool = True

    # VAE（LDM 风格 KL-VAE）
    vae_base_ch: int = 128
    vae_ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    vae_kl_weight: float = 1e-6
    # 训练时以算力换显存（encode_triplet 含三路 VAE，512 尺度易 OOM）
    vae_gradient_checkpointing: bool = True
    # v2 推荐：从已验证重建质量的 checkpoint 初始化 VAE 后冻结。
    freeze_vae: bool = False
    frozen_vae_use_mean: bool = True

    # 扩散（4.4.1）
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    # epsilon 与 v 均受支持；v-prediction 避免高噪声端从 epsilon
    # 还原 x0 时出现 1/sqrt(alpha_bar) 级误差放大。
    prediction_type: str = "epsilon"

    # 推理
    num_inference_steps: int = 50
    cfg_scale: float = 7.5

    # 损失权重（式 4-35 / 表 4-3）
    loss_w_low: float = 0.25
    loss_w_mid: float = 0.5
    loss_w_high: float = 1.0
    contrastive_weight: float = 0.1
    contrastive_tau: float = 0.07

    # 个体化微调（可选，式 4-33 / 4-34 / 表 4-3）
    personalization_token_dim: int = 256
    lambda_personalization: float = 1.0
    lambda_prior_preservation: float = 0.8

    def scale_weights(self) -> tuple[float, float, float]:
        return self.loss_w_low, self.loss_w_mid, self.loss_w_high
