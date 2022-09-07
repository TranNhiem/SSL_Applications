
import gradio as gr
import PIL
import os
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from torch import autocast
import torchcsprng as csprng
from stable_diffusion_inpainting import StableDiffusionInpaintingPipeline_
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
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

block = gr.Blocks(css=".container { max-width: 1200px; margin: auto; }")


def image_preprocess(image):
    image = Image.fromarray(image)
    w, h = image.size
    if w > 512:
        h = int(h * (512/w))
        w = 512
    if h > 512:
        w = int(w*(512/h))
        h = 512
    # resize to integer multiple of 64, 32 can sometimes result in tensor mismatch errors
    w, h = map(lambda x: x - x % 64, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    print(f"this is image.size: {image.size}")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def mask_processes(mask):
    mask = Image.fromarray(mask)
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


generator = csprng.create_random_device_generator('/dev/urandom')
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
    with torch.cuda.amp.autocast():
        images = pipeimg([prompt]*samples_num, init_image=img, mask_image=mask,
                         num_inference_steps=steps_num, guidance_scale=scale, generator=generator)["sample"]  # generator=generator
    return images


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
