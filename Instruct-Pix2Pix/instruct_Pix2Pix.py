## Tran Nhiem 2023/02/08 

'''

Implementation Pix2Pix & Pix2Pix Zero-Shot Model for Kid Education App 
    1. Pix2Pix 
    Pretrained Dataset 
    This Method using GPT3 to Generate text caption instructions --> then using Stable Diffusion + Image Prompt to prompt to 
    generate image pair with text caption instruction. --> create Large dataset on this --> Training diffusion model with this dataset

    2. Pix2Pix Zero-Shot
    without Pretraining Re-implemenation 
    - Using Cross Attention to Replace features of Image Similar Concept of "Prompt to Prompt"

    APP FEATURES : 
        1. Input: Image 
        2. Prompt instruction to change Image 
        3. Support Image Restore_UpScale, image color pallet.
        
        4. Support Prompt Suggestions for Prompt-to-Prompt (Fine-tune LLM to generate prompt suggestions)
        5. Support Multi-Language for Editting Image (Translation Module or Multi-Lingual Clip Model)

    
References: 
    
    1. https://www.timothybrooks.com/instruct-pix2pix/
    2. https://pix2pixzero.github.io/

'''

import os 
import sys 
import gradio as gr
import numpy as np
import cv2
import torch
import math
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline, DPMSolverMultistepScheduler
## Library for Image Color Pallet
from colorthief import ColorThief
## Upsacle and Restore Image with 
from codeformer_infer import inference as codeformer_inference

store_path="/data1/pretrained_weight/StableDiffusion/"
model_id = "timbrooks/instruct-pix2pix"

## Configure Scheduler for Shoter Inference Time 
DPM_Solver = DPMSolverMultistepScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        trained_betas=None,
        #predict_epsilon=True,
        thresholding=False,
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        lower_order_final=True,)


## InstructPix2Pix Pipeline
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                            model_id, 
                            torch_dtype=torch.float16, 
                            safety_checker=None, 
                            scheduler= DPM_Solver, 
                            ).to("cuda")
## Enable Flash Attention
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
# pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
pipe.enable_xformers_memory_efficient_attention()

## ------------------Hyperparameters-------------- 

help_text= '''

Two important hyperparameters:

text_CFG: Controls how much system follows text instructions. Higher value = more change based on text.

image_CFG: Controls similarity between input and output image. Higher value = more similarity to input.

Both values can be adjusted. Default is Image CFG 1.5 and Text CFG 7.5.

If image doesn't change enough:

    + Lower Image CFG
    + Raise Text CFG
If image changes too much:

    + Raise Image CFG
    + Lower Text CFG

'''

example_instructions = [
                        "Make it a picasso painting",
                        "Turn it into a dog image." , 
                        "Turn it into an anime.",
                        "have it look like a graphic novel.", 
                        ] 



steps=25
text_CFG=7.5
img_CFG=1.5
example_instructions ="Turn it into a cute cat image."



## Testing Code 
#
# edited_image = pipe(
#             example_instructions, 
#             image=input_image,
#             num_inference_steps=steps, 
#             guidance_scale=text_CFG, 
#             image_guidance_scale=img_CFG,
#             generator=generator,
#         ).images[0]

# edited_image.save("output_3.png")

## Building Class Object for InstructPix2Pix and Gradio Interface Web App
class InstructPix2Pix:
    def __init__(self, pipe, generator, steps=25, text_CFG=7.5, img_CFG=1.5):
        self.pipe = pipe
        self.generator = generator
        self.steps = steps
        self.text_CFG = text_CFG
        self.img_CFG = img_CFG

    def add_color_palette(self, image_path):
        
        ## Image shoud be str path 
        image = Image.open(image_path).convert("RGB")
        w, h= image.size 
        color_thief = ColorThief(image_path)
        colors = color_thief.get_palette(color_count=5, quality=1)
        height, width, channel=math.ceil(h/14), math.ceil(h/14), 3
        
        color=list(colors[0])
        red, green, blue= color[0], color[1], color[2]
        color_pallet=np.full((height, width, channel), [red, green, blue], dtype=np.uint8)
        for i in range(len(colors)): 
                if i ==0: 
                    continue
                else: 
                    color=colors[i]
                    color=list(color)
                    red, green, blue= color[0], color[1], color[2]
                    arr=np.full((height, width, channel), [red, green, blue], dtype=np.uint8)
                    color_pallet=cv2.hconcat([color_pallet, arr])

        color_pallet_pil=Image.fromarray(np.uint8(color_pallet)).convert('RGB')
        #image= Image.open(f"/home/harry/BLIRL/SSL_Applications/text_2_img/img{i_}.png")
        w_, h_= color_pallet_pil.size
        image.paste(color_pallet_pil, ( w-w_,h-h_))
        # palette = [tuple(color) for color in palette]
        return image

    def add_restore_upscale(self, image):
        '''Using CodeFormer to restore and upscale image''' 
        upscale_restore_img=codeformer_inference(image, background_enhance= True, face_upsample= True, upscale=4, codeformer_fidelity= 1.0, model_type="4x")
        return upscale_restore_img

    def __call__(self, image, instructions):
        edited_image = self.pipe(
            instructions, 
            image=image,
            num_inference_steps=self.steps, 
            guidance_scale=self.text_CFG, 
            image_guidance_scale=self.img_CFG,
            generator=self.generator,
        ).images[0]
        return edited_image 

## Gradio inference function 
def inference(image, instructions, seed=12340000):
    
    ## Image Preprocessing Reading & Resize Image 
    image = Image.fromarray(image.astype("uint8"), "RGB")
    width, height = image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    
    generator = torch.manual_seed(seed)
    image = InstructPix2Pix(pipe, generator)(image, instructions)
   
    ## Adding Restore and Upscale
    #image = InstructPix2Pix.add_restore_upscale(InstructPix2Pix, image)
    
    ## Adding Color Palette 
    image_path= image.save("/home/harry/BLIRL/SSL_Applications/Instruct-Pix2Pix/output_1.png")
    image_path="/home/harry/BLIRL/SSL_Applications/Instruct-Pix2Pix/output_1.png"
    image = InstructPix2Pix.add_color_palette(InstructPix2Pix, image_path)
    
    return image

## Gradio interface GUI 
def gradio_demo():
    gr.Interface(
        inference,
        [
            gr.inputs.Image( label="Input Image"),#shape=(512, 512),
            gr.inputs.Textbox(lines=2, label="Instructions"),
        ],
        gr.outputs.Image(type="pil", label="Output Image"),
        title="InstructPix2Pix",
        description=help_text,
        allow_flagging=False,
        examples=[
            [
                "/home/harry/BLIRL/SSL_Applications/Instruct-Pix2Pix/output_2.png",
                example_instructions,
            ]
        ],
    ).launch(share=True)
     
if __name__ == "__main__":
    gradio_demo()
