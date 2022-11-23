
# @TranNhiem 2022 11/05
'''
Features Design include:
1.. Creating MASK using brush to draw mask or creating mask using ClipSeg model () 
2.. Text to Image Inpainting (Mask area or Background area)
3.. Supporting multi-Language input 
4.. Prompting Using Language model for create prompt or User prompt input. 
5.. Creating Plugin exmaple style of Prompt and Upsampling for Inpainiting.

## Installations requirements 
!pip install -qq -U diffusers==0.6.0 transformers ftfy gradio
!pip install git+https://github.com/huggingface/diffusers.git

## Reference for Image Generation checkpoint model
Update Reference for Image Inpainting with Diffusion Models
https://huggingface.co/runwayml/stable-diffusion-inpainting 
https://huggingface.co/spaces/runwayml/stable-diffusion-inpainting/tree/main 


Note: You must comment these line of code from line 43 --> 49 of diffusers versio 0.6.0
    # mask = Image.fromarray(mask)
    # mask = np.array(mask.convert("L"))
    # mask = mask.astype(np.float32) / 255.0
    # mask = mask[None, None]
    # mask[mask < 0.5] = 0
    # mask[mask >= 0.5] = 1
    # mask = torch.from_numpy(mask)
   
"/anaconda/envs/solo_learn/lib/python3.10/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py"
'''
 
import os
import sys 
import random
import torch
import PIL
from PIL import Image
import numpy as np 
import gradio as gr
from torch import autocast
## Library for Stable diffusion inpainting model 
#from diffusion_pipeline.sd_inpainting_pipeline import  StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
from glob import glob
from diffusers import LMSDiscreteScheduler
from torchvision import transforms
## API for Language Translation Model 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


nllb_model= "/data1/pretrained_weight/NLLB"
SD_model="/data1/pretrained_weight/StableDiffusion"

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B",cache_dir=nllb_model )
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=nllb_model)
###--------------------------------
### Section for SD  Model 
###--------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
lms = LMSDiscreteScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
pipeimg = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    #"runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
    cache_dir=SD_model,
    #scheduler=lms,
    #use_auth_token=True,
).to("cuda")
pipeimg.scheduler = LMSDiscreteScheduler.from_config(pipeimg.scheduler.config)

def dummy(images, **kwargs): return images, False
pipeimg.safety_checker = dummy


generator = torch.Generator(device="cuda").manual_seed(random.randint(0,10000)) # change the seed to get different results

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

def infer(prompt_, samples_num, language_input,scale=7.5, steps_num=50, ): #option
    ## Checking the Image Size then Resize image

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

    ## Language Translation
    if source_langage[language_input] != "English" and source_langage[language_input] != "empty":
        translator_prompt = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_langage[language_input], tgt_lang='eng_Latn', max_length = 400)
        prompt= translator_prompt(prompt_)[0]
        prompt_=prompt['translation_text']
        print("Your English prompt translate from : ", prompt_)

        prompt_= [prompt_]*samples_num
    else:
        
        prompt_= [prompt_]*samples_num
        print(prompt_)
    # Generate image for the masking area with prompt
    with autocast("cuda"):#"cuda"
    # with torch.cuda.amp.autocast(dtype=torch.float16):
        images = pipeimg(prompt=prompt_,num_inference_steps=steps_num, guidance_scale=scale, generator=generator).images
        #breakpoint()
        return images

def run_demo(): 
    block = gr.Blocks(css=".container { max-width: 1300px; margin: auto; }")
    with block as demo:
        gr.HTML(read_content("header.html"))
        with gr.Group():
            with gr.Box():
                with gr.Row().style(mobile_collapse=False, equal_height=True):
                    with gr.Column(scale=4, min_width=200, min_height=600):
                        language_input = gr.Dropdown( ["🇱🇷 English", "🇻🇳 Vietnamese", "🇹🇼 TraditionalChinese", "🇨🇳 SimplifiedChinese", "🇫🇷 French", 
                        "🇩🇪 German","🇲🇨 Indonesian","🇯🇵 Japanese ","🇰🇷 Korean","🇪🇸 Spanish", "🇹🇭 Thai", ],label="🌎 Choosing Your Language: 🇱🇷,🇻🇳,🇹🇼,🇨🇳,🇫🇷,🇩🇪,🇯🇵 ", show_label=True)
                    
                    with gr.Column(scale=4, min_width=800, min_height=600):
                        text = gr.Textbox(label="Your text prompt", placeholder="Typing: (what you want to edit in your image)..", show_label=True, max_lines=1).style(
                            border=(True, False, True, True),
                            rounded=(True, False, False, True),
                            container=False,)
                 
                    #with gr.Row().style(mobile_collapse=False, equal_height=True):
                    with gr.Column(scale=4, min_width=800, min_height=600):
                        samples_num = gr.Slider(label="Number of Generated Image",
                                                minimum=1, maximum=10, value=4, step=1,)  # show_label=False


                    with gr.Column(scale=4, min_width=100, min_height=300):
                        btn = gr.Button("Run").style(
                            margin=False, rounded=(True, True, True, True),)


            with gr.Row().style(mobile_collapse=False,):#gallery

                with gr.Column():  #scale=1, min_width=80, min_height=300
                    gallery = gr.Gallery(label="Generated images",show_label=True).style(grid=[2], height="auto")

            btn.click(fn=infer, inputs=[text, samples_num, language_input, ], outputs=[gallery])

           
            gr.HTML(
                """
                <div class="footer">
                    <p style="align-items: center; margin-bottom: 7px;" >

                    </p>
                    <div style="text-align: Center; font-size: 1.5em; font-weight: bold; margin-bottom: 0.5em;">
                        <div style="
                            display: inline-flex; 
                            gap: 0.6rem; 
                            font-size: 1.0rem;
                            justify-content: center;
                            margin-bottom: 10px;
                            ">
                        <p style="align-items: center; margin-bottom: 7px;" >
                            App Developer: @TranNhiem 🙋‍♂️ Connect with me on : 
                        <a href="https://www.linkedin.com/feed/" style="text-decoration: underline;" target="_blank"> 🙌 Linkedin</a> ;  
                            <a href="https://twitter.com/TranRick2" style="text-decoration: underline;" target="_blank"> 🙌 Twitter</a> ; 
                            <a href="https://www.facebook.com/jean.tran.336" style="text-decoration: underline;" target="_blank"> 🙌 Facebook</a> 
                        </p>
                        </div>
                    </div>
                    <div style="
                        display: inline-flex; 
                        gap: 0.6rem; 
                        font-size: 1.0rem;
                        justify-content: center;
                        margin-bottom: 8px;
                        ">
                        </p> 
                        <p style="align-items: center; margin-bottom: 5px;" >
                        This App power Natural Language Translation and Text-To-Image Generation Model.
                        </p> 
                        <p>
                        1. Natural Language Translation Model power by NLLB-200
                        <a href="https://ai.facebook.com/research/no-language-left-behind/" style="text-decoration: underline;" target="_blank">NLLB</a>  
                        </p>
                        <p>
                        2. Text-to-Image generative model power by Stable Diffusion 
                        <a href="https://huggingface.co/CompVis" style="text-decoration: underline;" target="_blank">CompVis</a> and 
                        <a href="https://huggingface.co/stabilityai" style="text-decoration: underline;" target="_blank">Stability AI</a> 
                        </p>
                    </div>

                </div>
                <div class="acknowledgments">
                    <p><h4>LICENSE</h4>
                    The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> 
                    license. The authors claim no rights on the outputs you generate. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a>
                    </p>
                </div>
                """
            )
            
        
    demo.launch(share=True, enable_queue=True)  #server_name="172.17.0.1", # server_port=2222, share=True, enable_queue=True,  debug=True

if __name__ == '__main__':

    gr.close_all()
    run_demo()