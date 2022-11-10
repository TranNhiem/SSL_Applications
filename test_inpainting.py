import inspect
from typing import List, Optional, Union

import numpy as np
import torch

import PIL
import gradio as gr
#from diffusers import StableDiffusionInpaintPipeline
from diffusion_pipeline.sd_inpainting_pipeline import  StableDiffusionInpaintPipeline
from utils import mask_processes, image_preprocess
from diffusers import LMSDiscreteScheduler
from torch import autocast

device = "cuda"
model_path = "runwayml/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
).to(device)

prompt = "a mecha robot sitting on a bench"

guidance_scale=7.5
num_samples = 3
generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

def inpaint_image(img_mask, prompt, ):
    
    # if option == "Background Area":
    mask = img_mask['mask'].convert("RGB").resize((512, 512))
        #mask = mask_processes(img_mask['mask'])
    # else:
    #     #mask = 1- (img['mask']).convert("RGB")
    #     mask = 1 - mask_processes(img_mask['mask'])
    #     print("This is mask shape", mask.shape)

    #
    image=img_mask['image'].convert("RGB").resize((512, 512))
    #image = image_preprocess(img_mask['image'])
    with autocast("cuda"):
        images = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_samples,
        ).images

    return images[0]

gr.Interface(
    inpaint_image,
    title = 'Stable Diffusion In-Painting',
    inputs=[
        gr.Image(source = 'upload', tool = 'sketch', type = 'pil'),
        gr.Textbox(label = 'prompt')
    ],
    outputs = [ gr.Image()]
).launch(debug=True)