## TranNhiem 2022-09-02
'''
This script is used to test the performance of the stable diffusion inpainting algorithm.
Code is based on the code from HuggingFace's implementation of the stable diffusion inpainting algorithm.
https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb#scrollTo=cUBqX1sMsDR6 

# Usage: 
## 1. Scheduler Algorithm 
    + PNDM scheduler (used by default)
    + DDIM scheduler
    + K-LMS scheduler


'''

import numpy as np 
import torch 
import torchcsprng as csprng
from torch import autocast 
from diffusers import StableDiffusionPipeline, LSMDiscreteScheduler 
import requests 
import PIL 
from PIL import Image 
from io import BytesIO
from diffusers import AutoencoderKL, DDIMSscheduler, DiffusionPipeline, PNDMscheduler, UNet2DConditionModel 
from tqdm.auto import tqdm 
import inspect 
from typing import List, Tuple, Dict, Any, Optional, Union 
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer 

generator= csprng.creat_random_device_generator('/dev/urandom')

class StableDiffusionInpaintingPipeline(DiffusionPipeline): 
    def __init__(
        self, 
        vae: AutoencoderKL, 
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer, 
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMSscheduler, PNDMscheduler], 
        safety_checker: StableDiffusionSafetyChecker, 
        feature_extractor: CLIPFeatureExtractor,
    ): 
        super().__init__() 
        scheduler= scheduler.set_format("pt")
        self.register_modules(
            vae= vae, 
            text_encoder= text_encoder, 
            tokenizer= tokenizer, 
            unet = unet, 
            scheduler= scheduler, 
            safety_checker= safety_checker, 
            feature_extractor= feature_extractor,
        )
    @torch.no_grad()
    def __call(
        self, 
        prompt: Union[str, List[str]], 
        init_image: torch.FloatTensor, 
        mask_image: torch.FloatTensor, 
        num_inference_steps: Optional[int] = 50, 
        guidance_scale: Optional[float]= 7.5, 
        eta: Optional[float] = 0.0, 
        generator: Optional[torch.Generator] = None, 
        output_type: Optional[str] = "pil",
    ): 
        if isinstance(prompt, str): 
            batch_size=1 
        elif isinstance(prompt, list): 
            batch_size= len(prompt)
        else: 
            raise ValueError(f"prompt must be a string or a list of strings but U provided {type(prompt)}")

        ## Setting the timesteps 
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {} 
        offset = 0 
        if accepts_offset: 
            offset = 1 
            extra_set_kwargs["offset"] = 1
        
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # Encode the init image into latents and scale the latents 
        #     