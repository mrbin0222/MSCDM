from __future__ import annotations
import random
import torch
from torch import nn
from PIL import Image

class RandomQuarterTurn:

    def __call__(self, image: Image.Image) -> Image.Image:
        turns = random.randrange(4)
        if turns == 1:
            return image.transpose(Image.Transpose.ROTATE_90)
        if turns == 2:
            return image.transpose(Image.Transpose.ROTATE_180)
        if turns == 3:
            return image.transpose(Image.Transpose.ROTATE_270)
        return image

def configure_trainable_dino(model: nn.Module, last_blocks: int) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = list(model.encoder.layer)
    if not 1 <= last_blocks <= len(layers):
        raise ValueError(f'last_vision_blocks must be in [1,{len(layers)}], got {last_blocks}')
    for layer in layers[-last_blocks:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    for parameter in model.layernorm.parameters():
        parameter.requires_grad_(True)
    names = [name for (name, parameter) in model.named_parameters() if parameter.requires_grad]
    if not names:
        raise RuntimeError('no DINOv2 parameters are trainable')
    return names

def trainable_state(backbone: nn.Module, trainable_names: list[str]) -> dict[str, torch.Tensor]:
    parameters = dict(backbone.named_parameters())
    return {name: parameters[name].detach().cpu() for name in trainable_names}
