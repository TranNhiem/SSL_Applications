'''
TranNhiem 2023/02/24


    App Features Expected: 
        + Support Drawing 
        + Support Adding Multiple Language Input 
        + 
    Product Expect Service 
        + Kids Education Platform 
        + Architecture Design, Artistic 
        
Reference Building App 
    + https://github.com/cloneofsimo/paint-with-words-sd 


'''
import gradio as gr

from io import BytesIO
import requests
import PIL
from PIL import Image
import numpy as np
import os
import uuid
import torch
from torch import autocast
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from diffusers import DiffusionPipeline
from diffusers.utils import torch_device

store_path="/data1/pretrained_weight/StableDiffusion/"
pipe = DiffusionPipeline.from_pretrained(
    "Fantasy-Studio/Paint-by-Example",
    torch_dtype=torch.float16,
    cache_dir=store_path
)
pipe = pipe.to("cuda")

# from share_btn import community_icon_html, loading_icon_html, share_js

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

def predict(dict, reference, scale, seed, step):
    width,height=dict["image"].size
    if width<height:
        factor=width/512.0
        width=512
        height=int((height/factor)/8.0)*8

    else:
        factor=height/512.0
        height=512
        width=int((width/factor)/8.0)*8
    init_image = dict["image"].convert("RGB").resize((width,height))
    mask = dict["mask"].convert("RGB").resize((width,height))
    generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None
    output = pipe(
        image=init_image,
        mask_image=mask,
        example_image=reference,
        generator=generator,
        guidance_scale=scale,
        num_inference_steps=step,
    ).images[0]
    return output #, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


css = '''
.container {max-width: 1150px;margin: auto;padding-top: 1.5rem}
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
#mask_radio .gr-form{background:transparent; border: none}
#word_mask{margin-top: .75em !important}
#word_mask textarea:disabled{opacity: 0.3}
.footer {margin-bottom: 45px;margin-top: 35px;text-align: center;border-bottom: 1px solid #e5e5e5}
.footer>p {font-size: .8rem; display: inline-block; padding: 0 10px;transform: translateY(10px);background: white}
.dark .footer {border-color: #303030}
.dark .footer>p {background: #0b0f19}
.acknowledgments h4{margin: 1.25em 0 .25em 0;font-weight: bold;font-size: 115%}
#image_upload .touch-none{display: flex}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {
    display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
}
#share-btn {
    all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
}
#share-btn * {
    all: unset;
}
#share-btn-container div:nth-child(-n+2){
    width: auto !important;
    min-height: 0px !important;
}
#share-btn-container .wrap {
    display: none !important;
}
'''
example={}
# ref_dir='examples/reference'
# image_dir='examples/image'
# ref_list=[os.path.join(ref_dir,file) for file in os.listdir(ref_dir)]
# ref_list.sort()
# image_list=[os.path.join(image_dir,file) for file in os.listdir(image_dir)]
# image_list.sort()


image_blocks = gr.Blocks(css=css)
with image_blocks as demo:
    gr.HTML(read_content("header.html"))
    with gr.Group():
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Source Image")
                    reference = gr.Image(source='upload', elem_id="image_upload", type="pil", label="Reference Image")

                with gr.Column():
                    image_out = gr.Image(label="Output", elem_id="output-img").style(height=400)
                    guidance = gr.Slider(label="Guidance scale", value=5, maximum=15,interactive=True)
                    steps = gr.Slider(label="Steps", value=50, minimum=2, maximum=75, step=1,interactive=True)

                    seed = gr.Slider(0, 10000, label='Seed (0 = random)', value=0, step=1)

                    with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                        btn = gr.Button("Paint!").style(
                            margin=False,
                            rounded=(False, True, True, False),
                            full_width=True,
                        )
                    # with gr.Group(elem_id="share-btn-container"):
                    #     community_icon = gr.HTML(community_icon_html, visible=True)
                    #     loading_icon = gr.HTML(loading_icon_html, visible=True)
                    #     share_button = gr.Button("Share to community", elem_id="share-btn", visible=True)
            
            
            # with gr.Row():
            #     with gr.Column():
            #         gr.Examples(image_list, inputs=[image],label="Examples - Source Image",examples_per_page=12)
            #     with gr.Column():
            #         gr.Examples(ref_list, inputs=[reference],label="Examples - Reference Image",examples_per_page=12)
            
            btn.click(fn=predict, inputs=[image, reference, guidance, seed, steps], outputs=[image_out])
            # share_button.click(None, [], [], _js=share_js)



            gr.HTML(
                """
                    <div class="footer">
                        <p>Model by <a href="" style="text-decoration: underline;" target="_blank">Fantasy-Studio</a> - Gradio Demo by 🤗 Hugging Face
                        </p>
                    </div>
                    <div class="acknowledgments">
                        <p><h4>LICENSE</h4>
        The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                """
            )

image_blocks.launch()
