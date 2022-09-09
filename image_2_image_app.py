## This application building text-2-Image, Image-2-Image Generation Application 

from dataclasses import asdict
from torch import autocast
from PIL import Image
from typing import List
import os
import torch
import time

from utils import GeneratorConfig
from stable_diffusion_model import StableDiffusion_text_image_to_image_

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# from huggingface_hub import notebook_login
# notebook_login() ## Using for Colab or jupyter notebook
# Running your python script using
# huggingface-cli login
# CUDA_VISIBLE_DEVICES="0,1,2,3,4"
# Using the image inpainting pipeline

# device = torch.device(
#     "cuda") if torch.cuda.is_available() else torch.device("cpu")
class Imagegenerator: 
    def __init__(self,model: str="CompVis/stable-diffusion-v1-4"):
        '''Generate Image from prompt'''
        self.pipeimg = StableDiffusion_text_image_to_image_.from_pretrained(
        model, revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
        ).to("cuda")

    def generate(self, config: GeneratorConfig) -> Image: 
        """Generate Image from prompt"""
        config.prompt= [config.prompt]* config.num_images
        with torch.cuda.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            result= self.pipeimg(**asdict(config))
        return result

