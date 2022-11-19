
# @TranNhiem 2022 11/05
'''
Features Design include:
This build on V1 Version of Dreambooth
1.. Creating the Uploading Folder () 
2.. Integrate Automatic GUI interface (Inpainting, Super Resolution, Style Transfer, Dreambooth)
3.. Supporting multi-Language input 
4.. Prompting Style generation "creative prompt or User prompt input". 

'''

import gradio as gr
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

g_cuda = torch.Generator(device='cuda')
seed = 52362 #@param {type:"number"}
g_cuda.manual_seed(seed)
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

ouput_dir ="/data1/StableDiffusion/Dreambooth/pretrained/rick_v2"
pipe = StableDiffusionPipeline.from_pretrained(ouput_dir, scheduler= scheduler, torch_dtype=torch.float16,).to("cuda")

def inference(prompt, num_samples):
    with torch.autocast("cuda"), torch.inference_mode():
        all_images = [] 
        images = pipe(prompt, num_images_per_prompt=num_samples, generator=g_cuda, num_inference_steps=50, guidance_scale=7.5).images
        all_images.extend(images)
    return all_images

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="prompt")
            samples = gr.Slider(label="Samples",value=1)
            run = gr.Button(value="Run")
        with gr.Column():
            gallery = gr.Gallery(show_label=False)

    run.click(inference, inputs=[prompt,samples], outputs=gallery)
    gr.Examples([["a photo of sks rick riding a bicycle", 1,1]], [prompt,samples], gallery, inference, cache_examples=False)


demo.launch()
