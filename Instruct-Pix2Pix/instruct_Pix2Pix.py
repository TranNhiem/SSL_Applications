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
        (Support NLLB And mBART model many-to-one multilingual translation Module)

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
## Library for Language Translation Mododel 
## NLLB Model 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
## 

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
###--------------------------------
### Section Sequence2sequence Language Translation model
###--------------------------------
source_langage={
        "🇱🇷 English": "eng_Latn",
        "🇻🇳 Vietnamese": "vie_Latn", 
        "🇹🇼 TraditionalChinese": "zho_Hant",
        "🇨🇳 SimplifiedChinese": "zho_Hans",
        "🇫🇷 French" : "fra_Latn",
        "🇩🇪 German": "deu_Latn",
        "🇲🇨 Indonesian": "ind_Latn",
        "🇯🇵 Japanese": "jpn_Jpan",
        "🇰🇷 Korean": "kor_Hang", 
        "🇪🇸 Spanish": "spa_Latn", 
        "🇹🇭 Thai": "tha_Thai",
        "": "empty",
    }


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
def inference(image, instructions, seed=12340000, steps=25, text_CFG=7.5, img_CFG=1.5, color_palette=False, restore_upscale=False, language_input="🇱🇷 English"):
    ## Image Preprocessing Reading & Resize Image 
    image = Image.fromarray(image.astype("uint8"), "RGB")
    width, height = image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    
    ### Language Translation 
    if source_langage[language_input] != "eng_Latn":
        NLLB_path= "/data1/pretrained_weight/NLLB"
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B",cache_dir=NLLB_path )#cache_dir=NLLB_path
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=NLLB_path)
        print(f"source input language: {source_langage[language_input]}")
        translator_prompt = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_langage[language_input], tgt_lang='eng_Latn', max_length = 400)
        instructions= translator_prompt(instructions)[0]
        instructions=instructions['translation_text']
        #prompt_= [prompt_]*int(samples_num)
        del translator_prompt
        torch.cuda.empty_cache()
        print("Your English prompt translate from : ", instructions)

    else:
        print("Your Original English prompt: ", instructions)
        



    generator = torch.manual_seed(seed)
    image = InstructPix2Pix(pipe, generator,steps=steps, text_CFG=text_CFG, img_CFG=img_CFG )(image, instructions)
   
    ## Adding Restore and Upscale
    if restore_upscale:
        image_path= image.save("/home/harry/BLIRL/SSL_Applications/Instruct-Pix2Pix/output_1.png")
        image_path="/home/harry/BLIRL/SSL_Applications/Instruct-Pix2Pix/output_1.png"
        image = InstructPix2Pix.add_restore_upscale(InstructPix2Pix, image_path)
    
    ## Adding Color Palette
    if color_palette:
        image_path= image.save("/home/harry/BLIRL/SSL_Applications/Instruct-Pix2Pix/output_1.png")
        image_path="/home/harry/BLIRL/SSL_Applications/Instruct-Pix2Pix/output_1.png"
        image = InstructPix2Pix.add_color_palette(InstructPix2Pix, image_path)
    
    # return image ## using this for simple_demo
    return [image]

## Simple Gradio interface GUI 
def simple_demo():
    gr.Interface(
        inference,
        [
            gr.inputs.Image( label="Input Image"),#shape=(512, 512),
            gr.inputs.Textbox(lines=2, label="Instructions"),
        ],

        #gr.outputs.Image(type="pil", label="Output Image"),
        gr.Gallery(label="Edited images",show_label=True).style(grid=[2], height="auto").style(height=400),

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
     
## Adapt Author GUI 
def read_content(file_path) :
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

def demo_2(): 
    block = gr.Blocks(css=".container { max-width: 1300px; margin: auto; }")
    with block as demo:
        gr.HTML(read_content("header.html"))
        with gr.Group():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                    with gr.Column(scale=4, min_width=200, min_height=600):
                        language_input = gr.Dropdown( ["🇱🇷 English", "🇻🇳 Vietnamese", "🇹🇼 TraditionalChinese", "🇨🇳 SimplifiedChinese", "🇫🇷 French", 
                        "🇩🇪 German","🇲🇨 Indonesian","🇯🇵 Japanese ","🇰🇷 Korean","🇪🇸 Spanish", "🇹🇭 Thai", ],value="🇱🇷 English", label="🌎 Choosing Your Language: 🇱🇷,🇻🇳,🇹🇼,🇨🇳,🇫🇷,🇩🇪,🇯🇵 ", show_label=True)
                    with gr.Column(scale=4, min_width=700, min_height=600):
                        instruction = gr.Textbox(label="Your text prompt", placeholder="Typing: (what you want to edit in your image)..", show_label=True,lines=2, max_lines=3).style(
                            border=(True, False, True, True),
                            rounded=(True, False, False, True),
                            container=False,)
                    with gr.Column(scale=4, min_width=100, min_height=300):
                        with gr.Row().style(mobile_collapse=False, equal_height=True):
                            btn = gr.Button("Generate")#.style(margin=False, rounded=(True, True, True, True),)
                        with gr.Row().style(mobile_collapse=False, equal_height=True):
                            stop_run = gr.Button("STOP")#.style(margin=False, rounded=(True, True, True, True),)

            with gr.Row().style(mobile_collapse=False,):#gallery

                with gr.Column():#scale=1, min_width=100, min_height=400
                    with gr.Row():
                        with gr.Group():
                            image= gr.inputs.Image( label="Input Image")
                            #input_image_properties = gr.Textbox(label="Image Properties", max_lines=1)
                    with gr.Row():
                        
                        with gr.Accordion("Advanced options", open=True):
                            ## Configuration Hyperparameters for Generation 
                            with gr.Row().style(mobile_collapse=False, equal_height=True):
                                with gr.Column(scale=4, min_width=300, min_height=600):
                                    text_cfg_scale = gr.Number(value=7.5, label=f"Text CFG", interactive=True)
                                with gr.Column(scale=4, min_width=300, min_height=600):
                                    image_cfg_scale = gr.Number(value=1.5, label=f"Image CFG", interactive=True)
                                with gr.Column(scale=4, min_width=200, min_height=600):
                                    steps = gr.Number(value=25, precision=0, label="Steps of Generation", interactive=True)
                                with gr.Column(scale=4, min_width=200, min_height=600):
                                    Seed =  gr.Number(value=1371, precision=0, label="Randomize Seed", interactive=True)
                            
                            ## Configuration the Options 
                            with gr.Row().style(mobile_collapse=False, equal_height=True):
                                with gr.Column(scale=4, min_width=300, min_height=600):
                                    upscale = gr.Checkbox( value=False, label="Upscale & Enhancement Image", show_label=True,)  # show_label=False
                                with gr.Column(scale=4, min_width=300, min_height=600):
                                    pallet_color = gr.Checkbox( value=True, label="Color Pallettes", show_label=True)

                with gr.Column():  #scale=1, min_width=80, min_height=300
                    
                    gallery = gr.Gallery(label="Edited images",show_label=True).style(grid=[2], height="auto").style(height=400)
                    #description=help_text
                    #<div class="footer">
                    gr.HTML(
                        """
                        
                      <div style="display: flex; flex-direction: row;">
                        <div style="flex: 1; margin-right: 20px;">
                            <p>1. Two important hyperparameters:</p>
                            <ul>
                            <li>text_CFG: Controls how much system follows text instructions. Higher value = more change based on text.</li>
                            <li>image_CFG: Controls similarity between input and output image. Higher value = more similarity to input.</li>
                            </ul>
                            <p>2. Upscaling & restore image -> good human face images checkbox "Upscale & Enhancement Image".</p>
                            <p>3. including color palettes -> checkbox "Color Pallettes" .</p>
                        </div>
                        <div style="flex: 1; margin-left: 20px;">
                            <p>Both values can be adjusted. Default is Image CFG 1.5 and Text CFG 7.5.</p>
                            <p>If image doesn't change enough:</p>
                            <ul>
                            <li>Lower Image CFG</li>
                            <li>Raise Text CFG</li>
                            </ul>
                            <p>If image changes too much:</p>
                            <ul>
                            <li>Raise Image CFG</li>
                            <li>Lower Text CFG</li>
                            </ul>
                        </div>
                        </div>

                        """)
            
            run_event=btn.click(fn=inference, inputs=[image,instruction, Seed, steps, text_cfg_scale, image_cfg_scale, pallet_color, upscale, language_input], outputs=[gallery])
            stop_run.click(fn=None, inputs=None, outputs=None, cancels=[run_event])
    
    demo.launch( share=True, enable_queue=True,  debug=True)  #server_name="172.17.0.1", # server_port=2222, share=True, enable_queue=True,  debug=True


if __name__ == "__main__":
    #simple_demo
    demo_2()
