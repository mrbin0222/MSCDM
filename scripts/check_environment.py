#!/usr/bin/env python3
import importlib.util,json
modules=['torch','torchvision','transformers','numpy','PIL','scipy','skimage','torchmetrics','torch_fidelity','tifffile','cellpose']
print(json.dumps({m:importlib.util.find_spec(m) is not None for m in modules},indent=2))
