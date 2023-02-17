# import math
# import os

# # import pytorch_lightning as pl
# import torch
# import torch.nn as nn
# # from pytorch_lightning import Trainer, seed_everything
# # from pytorch_lightning.utilities import rank_zero_info
# from torch.nn import functional as F
# from torch.utils.data import DataLoader, Dataset, RandomSampler

# from xformers.factory.model_factory import xFormer, xFormerConfig

import torch
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp, MemoryEfficientAttentionOp

store_path="/data1/pretrained_weight/StableDiffusion/"
pipe = DiffusionPipeline.from_pretrained(
                                        "runwayml/stable-diffusion-v1-5",
                                        # "stabilityai/stable-diffusion-2-1", 
                                         torch_dtype=torch.float16, 
                                         cache_dir=store_path,
                                        )

pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
# Workaround for not accepting attention shape using VAE for Flash Attention
pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)


# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch.float16,
#     cache_dir=store_path,
# ).to("cuda")

# pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)

# with torch.inference_mode():
#     sample = pipe("a small cat")
