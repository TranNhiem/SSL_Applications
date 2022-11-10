# @TranNhiem 2022 09/05
'''
Update Reference for Image Inpainting with Diffusion Models
+ Runway model 
https://huggingface.co/runwayml/stable-diffusion-inpainting 
https://huggingface.co/spaces/runwayml/stable-diffusion-inpainting/tree/main 

'''


import gradio as gr

import os
import sys 
import random
import torch
from torch import autocast
# import torchcsprng as csprng
from stable_diffusion_model import  StableDiffusionInpaintingPipeline_
from diffusers.pipelines.stable_diffusion import StableDiffusionInpaintPipeline

from glob import glob
from utils import mask_processes, image_preprocess

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# from huggingface_hub import notebook_login
# notebook_login() ## Using for Colab or jupyter notebook
# Running your python script on terminal or cmd
# huggingface-cli login
# CUDA_VISIBLE_DEVICES="0,1,2,3,4"
# Using the image inpainting pipeline

# device = torch.device(
#     "cuda") if torch.cuda.is_available() else torch.device("cpu")
lms = LMSDiscreteScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

pipeimg = StableDiffusionInpaintPipeline.from_pretrained(
    #"CompVis/stable-diffusion-v1-4",
    "runwayml/stable-diffusion-v1-5", 
    #"runwayml/stable-diffusion-inpainting",
     revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
    scheduler=lms,
).to("cuda")


def dummy(images, **kwargs): return images, False

pipeimg.safety_checker = dummy

generator = torch.Generator(device="cuda").manual_seed(random.randint(0,10000)) # change the seed to get different results
# generator = csprng.create_random_device_generator('/dev/urandom')
example_dir = "/home/rick/code_spaces/SSL_Applications/Bird_images"
# example_dir="Bird_images"


def infer(prompt, img, samples_num, steps_num, scale, option):

    if option == "Background Area":
        mask = (img['mask'])#.convert("RGB")
        #mask = mask_processes(img['mask'])
    else:
        mask = (img['mask'])#.convert("RGB")
        #mask = 1 - mask_processes(img['mask'])

    #
    image=(img['image'])#.convert("RGB")
    #image = image_preprocess(img['image'])

    print(prompt)
    # Generate image for the masking area with prompt
    # with autocast("cuda"):#"cuda"
    with torch.cuda.amp.autocast(dtype=torch.float16):
        images = pipeimg([prompt]*samples_num, image=image, mask_image=mask,
                         num_inference_steps=steps_num, guidance_scale=scale, generator=generator)["sample"]  # generator=generator
    return images

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

def run_demo(): 
    block = gr.Blocks(css=".container { max-width: 1300px; margin: auto; }")
    with block as demo:
        gr.Markdown(
            "<h1><center> Image Inpainting 🎨 - 🖼️ </center></h1>")
        gr.HTML(
                    """
                        <div class="footer">
                            <p> INSTRUCTION to USE this App <a href=" https://youtu.be/q5kAOi-edoY" style="text-decoration: underline;" target="_blank"> Video Link </a> 
                            </p>
                        </div> 
                    """
                )
        with gr.Group():
            with gr.Box():
                with gr.Row().style(mobile_collapse=False, equal_height=True):
                    with gr.Column(scale=4, min_width=900, min_height=600):
                        text = gr.Textbox(label="Inpainting with Your text prompt", placeholder="Enter In-painting expected object here...", show_label=True, max_lines=1).style(
                            border=(True, False, True, True),
                            rounded=(True, False, False, True),
                            container=False,)
                    with gr.Column(scale=4, min_width=150, min_height=600):
                        btn = gr.Button("Run").style(
                            margin=False, rounded=(True, True, True, True),)

            with gr.Row().style(mobile_collapse=False, equal_height=True):
                samples_num = gr.Slider(label="Number of Generated Image",
                                        minimum=1, maximum=10, value=1, step=1,)  # show_label=False
                steps_num = gr.Slider(
                    label="Generatio of steps", minimum=2, maximum=499, value=80, step=1,)  # show_label=False
                scale = gr.Slider(label="Guidance scale", minimum=0.0,
                                maximum=30, value=7.5, step=0.1,)  # show_label=False

            with gr.Row().style(mobile_collapse=False, equal_height=True):
                # option = gr.Radio(label=" Selecting Inpainting Area", default="Mask Area", choices=[
                #     "Mask Area", "Background Area"], show_label=True)
                option = gr.Dropdown( ["Replacing Mask Area", "Replacing Background Area"],label=" Choosing Replacing Part", show_label=True)
            
            image = gr.Image(source="upload", tool="sketch",label="Input image", type="pil").style(height=400)
            gallery = gr.Gallery(label="Generated images",show_label=True).style(grid=[2], height="auto")
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

    demo.launch(server_name="0.0.0.0",  server_port=123456,
                share=True, enable_queue=True, )  # debug=True)
if __name__ == '__main__':

    gr.close_all()
    run_demo()