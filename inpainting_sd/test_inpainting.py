# @TranNhiem 2022 11/05
'''
Features Design include:
1.. Creating MASK using brush to draw mask or creating mask using ClipSeg model () 
2.. Text to Image Inpainting (Mask area or Background area)
3.. Prompting Using Language model for create prompt or User prompt input. 
4.. Creating Plugin exmaple style of Prompt and Upsampling for Inpainiting.

## Installations requirements 
!pip install -qq -U diffusers==0.6.0 transformers ftfy gradio
!pip install git+https://github.com/huggingface/diffusers.git

## Reference for Image Generation checkpoint model
Update Reference for Image Inpainting with Diffusion Models
https://huggingface.co/runwayml/stable-diffusion-inpainting 
https://huggingface.co/spaces/runwayml/stable-diffusion-inpainting/tree/main 

'''

import inspect
from typing import List, Optional, Union
from PIL import Image
import numpy as np
import torch
import PIL
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
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

def mask_processes(mask):
    mask = mask.convert("L")
    w, h = mask.size
    if w > 512:
        h = int(h * (512/w))
        w = 512
    if h > 512:
        w = int(w*(512/h))
        h = 512
    w, h = map(lambda x: x - x % 64, (w, h))
    w //= 8
    h //= 8

    mask = mask.resize((w, h), resample=PIL.Image.LANCZOS)
    print(f"Mask size:, {mask.size}")
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)
    mask[np.where(mask != 0.0)] = 1.0  # using bool to find the uer drawed
    mask = torch.from_numpy(mask)
    return mask



def inpaint_image(img_mask, prompt,painting_option, guidance_scale, num_samples):
    ## Building mask 
    if painting_option == "Background Area": 
        mask = mask_processes(img_mask['mask'])
        mask=mask.convert("RGB").resize((512, 512))
    else: 
        mask = 1 - mask_processes(img_mask['mask'])
        mask=mask.convert("RGB").resize((512, 512))

 
    image=img_mask['image'].convert("RGB").resize((512, 512))
    #image = image_preprocess(img_mask['image'])
    with autocast("cuda"):
        images = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_samples,).images
    return images[0]

gr.Interface(
    inpaint_image,
    title = 'Stable Diffusion In-Painting',
    inputs=[gr.Image(source = 'upload', tool = 'sketch', type = 'pil'),
            gr.Textbox(label = 'prompt')
    ],
    outputs = [ gr.Image()]
).launch(debug=True)