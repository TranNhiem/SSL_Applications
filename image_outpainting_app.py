
# @TranNhiem 2022/09/06 
import gradio as gr
from PIL import Image, ImageDraw
import math 

## Function to get the adding Area
def ui():
    pixels = gr.Slider(label="Pixels to expand", minimum=8, maximum=128, step=8)
    mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, visible=False)
    inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='fill', type="index", visible=False)

    return [pixels, mask_blur, inpainting_fill]

