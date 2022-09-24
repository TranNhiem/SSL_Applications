# @TranNhiem 2022 09/05
import gradio as gr

import os
import sys 
import random
import torch
from torch import autocast
# import torchcsprng as csprng
from stable_diffusion_model import StableDiffusionInpaintingPipeline_
from glob import glob
from utils import mask_processes, image_preprocess

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# from huggingface_hub import notebook_login
# notebook_login() ## Using for Colab or jupyter notebook
# Running your python script using
# huggingface-cli login
# CUDA_VISIBLE_DEVICES="0,1,2,3,4"
# Using the image inpainting pipeline

# device = torch.device(
#     "cuda") if torch.cuda.is_available() else torch.device("cpu")

pipeimg = StableDiffusionInpaintingPipeline_.from_pretrained(
    "CompVis/stable-diffusion-v1-4", revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
).to("cuda")


def dummy(images, **kwargs): return images, False

pipeimg.safety_checker = dummy

generator = torch.Generator(device="cuda").manual_seed(random.randint(0,10000)) # change the seed to get different results
# generator = csprng.create_random_device_generator('/dev/urandom')
example_dir = "/home/rick/code_spaces/SSL_Applications/Bird_images"
# example_dir="Bird_images"


def infer(prompt, img, samples_num, steps_num, scale, option):

    if option == "Mask Sketch Area":
        mask = 1 - mask_processes(img['mask'])
    else:
        mask = mask_processes(img['mask'])
    img = image_preprocess(img['image'])
    print(prompt)
    # Generate image for the masking area with prompt
    # with autocast("cuda"):#"cuda"
    with torch.cuda.amp.autocast(dtype=torch.float16):
        images = pipeimg([prompt]*samples_num, init_image=img, mask_image=mask,
                         num_inference_steps=steps_num, guidance_scale=scale, generator=generator)["sample"]  # generator=generator
    return images

block = gr.Blocks(css=".container { max-width: 1300px; margin: auto; }")
with block as demo:
    gr.Markdown(
        "<h1><center> Image Inpainting App </center></h1> Different image resolutions should be 'working'")

    gr.Markdown(
        "<h3><center> Illustration How to use this App </center></h3> https://youtu.be/q5kAOi-edoY ")

    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(label="Inpainting with Your prompt", placeholder="Enter In-painting expected object here...", show_label=True, max_lines=1).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,)
                btn = gr.Button("Run").style(
                    margin=True, rounded=(True, True, True, True),)

        with gr.Row().style(mobile_collapse=False, equal_height=True):
            samples_num = gr.Slider(label="Number of Generated Image",
                                    minimum=1, maximum=10, value=1, step=1,)  # show_label=False
            steps_num = gr.Slider(
                label="Generatio of steps", minimum=2, maximum=499, value=80, step=1,)  # show_label=False
            scale = gr.Slider(label="Guidance scale", minimum=0.0,
                              maximum=30, value=7.5, step=0.1,)  # show_label=False

        with gr.Row().style(mobile_collapse=False, equal_height=True):
            option = gr.Radio(label="Inpainting Area", default="Mask Sketch Area", choices=[
                "Mask Sketch Area", "Background Area"], show_label=True)

        image = gr.Image(source="upload", tool="sketch",
                         label="Input image", type="numpy")
        gallery = gr.Gallery(label="Generated images",
                             show_label=True).style(grid=[2], height="auto")
        text.submit(fn=infer, inputs=[
                    text, image, samples_num, steps_num, scale, option], outputs=gallery)

        btn.click(fn=infer, inputs=[text, image, samples_num,
                  steps_num, scale, option], outputs=gallery)

        # with gr.Row("20%"):
        #      gr.Markdown("#### (Illustration of Using This App):")
        #      gr.Image(f"{example_dir}/Rotating_earth_(large).gif", shape=(224, 224))

        # Using example images provide
        # with gr.Row("20%"):
        #     gr.Markdown("#### (Image Example):")
        #     #ims_uri = [f"{example_dir}/{x}" for x in os.listdir(example_dir)]##
        #     ims_uri=f"{example_dir}/343785.jpg"
        #     # ims_uri = [ex for ex in ims_uri]

        #     image = gr.Image(tool="sketch",label="Input image", type="numpy", value=str(ims_uri))#value=f"{example_dir}/343785.jpg") #source='upload',  # value=os.path.join(os.path.dirname(__file__),".jpg"
        #     gr.Examples(fn=infer, examples=[os.path.join(os.path.dirname(__file__)),image], inputs=[text, image, samples_num,steps_num, scale, option], outputs=gallery, cache_examples=True)

demo.launch(server_name="140.115.53.102",  server_port=1111,
            share=True, enable_queue=True, )  # debug=True)
