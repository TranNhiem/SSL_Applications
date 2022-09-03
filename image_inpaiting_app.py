
import gradio as gr
import PIL
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from torch import autocast
import torchcsprng as csprng
from stable_diffusion_inpainting import StableDiffusionInpaintingPipeline

# from huggingface_hub import notebook_login
# notebook_login() ## Using for Colab or jupyter notebook
# Running your python script using
# huggingface-cli login

# Using the image inpainting pipeline
pipeimg = StableDiffusionInpaintingPipeline.from_pretrained(
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


def infer(prompt, img, samples_num, steps_num, scale, option):

    if option == "Replace selection":
        mask = 1 - mask_processes(img['mask'])
    else:
        mask = mask_processes(img['mask'])
    img = image_preprocess(img['image'])

    # Generate image for the masking area with prompt
    with autocast("cuda"):
        images = pipeimg([prompt]*samples_num, img, mask,
                         num_inference_steps=steps_num, guidance_scale=scale,  generator=generator)

    return images


with block as demo:
    gr.Markdown(
        "<h1><center> Image Inpainting App </center></h1>arbitrary resolutions should be 'working'")
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(label="Inpainting with Your prompt", show_label=False, max_lines=10).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Run").style(margin=False,
                                             rounded=(False, True, True, False),)

            with gr.Row().style(mobile_collapse=False, equal_height=True):
                samples_num = gr.Slider(
                    label="Number of samples", min=1, max=4, value=1, step=1,)  # show_label=False
                steps_num = gr.Slider(
                    label="Generatio of steps", min=2, max=499, value=80, step=1,)  # show_label=False
                scale = gr.Slider(label="Guidance scale", min=0.0,
                                  max=30, value=7.5, step=0.1,)  # show_label=False

            with gr.Row().style(mobile_collapse=False, equal_height=True):
                option = gr.Radio(label="Inpainting option", choices=[
                                  "Replace selection", "Inpaint outside selection"], show_label=False)

            image = gr.Image(source='upload', tool="sketch",
                             label="Input image", type="numpy")

            gallery = gr.Gallery(label="Generated images", show_label=False).style(
                grid=[2], height="auto"
            )

            text.submit(infer, inputs=[
                        text, image, samples_num, steps_num, scale, option], outputs=gallery)
            btn.click(infer, inputs=[
                      text, image, samples_num, steps_num, scale, option], outputs=gallery)

demo.launch(server_name="0.0.0.0",  server_port=5555,
            share=False, enable_queue=True, )  # debug=True)
