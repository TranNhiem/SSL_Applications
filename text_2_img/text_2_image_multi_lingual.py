
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
import gradio as gr, re

from torch import autocast
## Library for Stable diffusion inpainting model 
#from diffusion_pipeline.sd_inpainting_pipeline import  StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
import random
from glob import glob
from diffusers import LMSDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from torchvision import transforms
## API for Language Translation Model 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, GPT2LMHeadModel, GPT2Tokenizer

from colorthief import ColorThief
from pathlib import Path
import cv2
import math
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

###--------------------------------
### Section for Translation Model
###--------------------------------

nllb_model= "/data1/pretrained_weight/NLLB"
SD_model="/data1/pretrained_weight/StableDiffusion"

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B",cache_dir=nllb_model )
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=nllb_model)
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
      
    }

###--------------------------------
### Section Magic Prompt Generation
###--------------------------------
prompt_gen_model="/data1/pretrained_weight/prompt_gen"
# only cache the latest model
def get_model_and_tokenizer(model_id):
    model = GPT2LMHeadModel.from_pretrained(model_id, cache_dir=prompt_gen_model, device_map="auto") ## [load_in_8bit=True, torch_dtype=torch.float16]
    tokenizer = GPT2Tokenizer.from_pretrained(model_id,cache_dir=prompt_gen_model)
    return model, tokenizer


###--------------------------------
### Section for SD  Model 
###--------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def dummy(images, **kwargs): return images, False

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

def prompt_magic(prompt_,language_input ):
    ## Generate the Prompt with initial Prompt
    model_id= "Gustavosta/MagicPrompt-Stable-Diffusion"
    model, tokenizer =get_model_and_tokenizer(model_id)
    print(source_langage[language_input])
    if source_langage[language_input] != "🇱🇷 English":
        translator_prompt = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_langage[language_input], tgt_lang='eng_Latn', max_length = 400)
        prompt= translator_prompt(prompt_)[0]
        prompt_=prompt['translation_text']
        print(prompt_)
    
    input_ids = tokenizer(prompt_, return_tensors='pt').input_ids
    

    # generate the result with contrastive search
    contrastive_sampling = model.generate(input_ids, penalty_alpha=0.6, top_k=4, max_length=random.randint(60, 90), num_return_sequences=1)
    nucleus_sampling = model.generate(input_ids, do_sample=True, top_p=0.95, top_k=0, max_length=random.randint(60, 90), num_return_sequences=1)
    # beam_sampling=model.generate(input_ids, do_sample=True, num_beams=4, max_length=random.randint(60, 90), num_return_sequences=1)#random.randint(60, 90)
    # greedy_sampling=model.generate(input_ids, do_sample=True,  max_length=random.randint(60, 90), num_return_sequences=1)

    contrastive_text = str("1. ")+ tokenizer.decode(contrastive_sampling[0], skip_special_tokens=True)
    nucleus_text = str("2. ")+ tokenizer.decode(nucleus_sampling[0], skip_special_tokens=True)
    # beam_text = str("3. ")+ tokenizer.decode(beam_sampling[0], skip_special_tokens=True)
    # greedy_text = str("4. ")+ tokenizer.decode(greedy_sampling[0], skip_special_tokens=True)

    if source_langage[language_input] != "🇱🇷 English":
        translator_prompt = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang=source_langage[language_input], max_length = 400)
        
        contrastive_text=translator_prompt(contrastive_text)[0]
        contrastive_text=contrastive_text['translation_text']
        nucleus_text=translator_prompt(nucleus_text)[0]
        nucleus_text=nucleus_text['translation_text']
        # beam_text=translator_prompt(beam_text)[0]
        # beam_text=beam_text['translation_text']
        # greedy_text=translator_prompt(greedy_text)[0]
        # greedy_text=greedy_text['translation_text']
    

    response_list=[contrastive_text, nucleus_text,] # beam_text, greedy_text
    response_end = "\n".join(response_list)
    response_end = re.sub('[^ ]+\.[^ ]+','', response_end)
    response_end = response_end.replace("<", "").replace(">", "")

    return response_end
    
def infer(prompt_, language_input, model_id, pallet_color=True, samples_num=4, seed=123455, steps_num=25,scale=7.5,): #option
    ## Checking the Image Size then Resize image

    generator = torch.Generator(device="cuda").manual_seed(seed) # random.randint(0,10000)


    ## Language Translation
    if source_langage[language_input] != "🇱🇷 English":
        translator_prompt = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_langage[language_input], tgt_lang='eng_Latn', max_length = 400)
        prompt= translator_prompt(prompt_)[0]
        prompt_=prompt['translation_text']
        print("Your English prompt translate from : ", prompt_)
        prompt_= [prompt_]*samples_num
    else:
        
        prompt_= [prompt_]*samples_num
        print(prompt_)



    model_id_={
        "Artistic": "prompthero/openjourney",
        "Animation":  "hakurei/waifu-diffusion", 
        "Realistic_v1": "CompVis/stable-diffusion-v1-4",
        "Realistic_v2": "runwayml/stable-diffusion-v1-5",
       
        #"Model-4": "stabilityai/stable-diffusion-2", 
    }
    # Generate image for the masking area with prompt
    SD_model="/data1/pretrained_weight/StableDiffusion/"
    LMSD = LMSDiscreteScheduler.from_config(model_id_[model_id], subfolder="scheduler")
    
    Euler = EulerDiscreteScheduler.from_pretrained(model_id_[model_id], subfolder="scheduler", prediction_type="v_prediction")
    DPM_Solver = DPMSolverMultistepScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        trained_betas=None,
        predict_epsilon=True,
        thresholding=False,
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        lower_order_final=True,)
    
    pipeimg = StableDiffusionPipeline.from_pretrained(
        #"CompVis/stable-diffusion-v1-4",
        model_id_[model_id],
        #SD_model,
        #"runwayml/stable-diffusion-inpainting",
        #revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=SD_model,
        scheduler=DPM_Solver,
        #use_auth_token=True,
    ).to("cuda")
    #pipeimg.enable_xformers_memory_efficient_attention()
    pipeimg.safety_checker = dummy

    # pipeimg.scheduler = LMSDiscreteScheduler.from_config(pipeimg.scheduler.config)

    # Generate image for the masking area with prompt

    with autocast("cuda"):#"cuda"
    # with torch.cuda.amp.autocast(dtype=torch.float16):
        images = pipeimg(prompt=prompt_,num_inference_steps=steps_num, guidance_scale=scale, generator=generator).images
        
    
    if pallet_color:
        
        print("*************Adding Pallet Color****************")
        w, h= images[0].size
        for id, img in enumerate(images): 
            img.save(f"/home/harry/BLIRL/SSL_Applications/text_2_img/img{id}.png")
        
        images_=[]
        for i_ in range(len(images)):
            color_thief = ColorThief(f"/home/harry/BLIRL/SSL_Applications/text_2_img/img{i_}.png")
            colors = color_thief.get_palette(color_count=5, quality=1)
            #colors_hex = ['#%02x%02x%02x' % (color) for color in colors]
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
            image= Image.open(f"/home/harry/BLIRL/SSL_Applications/text_2_img/img{i_}.png")
            w_, h_= color_pallet_pil.size
            image.paste(color_pallet_pil, ( w-w_,h-h_))
            
            images_.append(image)
        images=images_

    return images

    

examples = [
    [
        'Photo of two black swans touching heads in a beautiful reflective mountain lake, a colorful hot air balloon is flying above the swans, hot air balloon, intricate, 8k highly professionally detailed, hdr, cgsociety',"🇱🇷 English", "Artistic"
        ],
    ['Hình ảnh hai con thiên nga đen chạm đầu nhau trong một hồ nước trên núi phản chiếu tuyệt đẹp, một chiếc khinh khí cầu đầy màu sắc đang bay phía trên những con thiên nga, khinh khí cầu, 8k rất chi tiết chuyên nghiệp, hdr, cgsociety',"🇻🇳 Vietnamese",  "Artistic" ], 
    [
        '兩隻黑天鵝在美麗的反光山湖中碰頭的照片，一個彩色熱氣球在天鵝上方飛行，熱氣球，錯綜複雜，8k 高度專業詳細，hdr，cgsociety', "🇹🇼 TraditionalChinese", "Artistic",
        ],
    [
        '两只黑天鹅在美丽的反光山湖中碰头的照片，一个彩色热气球在天鹅上方飞行，热气球，错综复杂，8k 高度专业详细，hdr，cgsociety', "🇨🇳 SimplifiedChinese","Artistic", 
        ],

    [
        "Photo de deux cygnes noirs touchant la tête dans un magnifique lac de montagne réfléchissant, une montgolfière colorée vole au-dessus des cygnes, montgolfière, complexe, 8k très professionnellement détaillée, hdr, cgsociety",  "🇫🇷 French", "Artistic"
        ],
    [
        "Foto von zwei schwarzen Schwänen, die Köpfe in einem wunderschönen reflektierenden Bergsee berühren, ein bunter Heißluftballon fliegt über den Schwänen, Heißluftballon, kompliziert, 8k hochprofessionell detailliert, hdr, cgsociety", "🇩🇪 German", "Artistic"
        ],
    [
        "Foto dua angsa hitam menyentuh kepala di danau gunung reflektif yang indah, balon udara panas berwarna-warni terbang di atas angsa, balon udara panas, rumit, detail 8k sangat profesional, hdr, cgsociety","🇲🇨 Indonesian", "Artistic"
    ],
    [
        "美しい反射する山の湖で頭に触れる2羽の黒い白鳥の写真、カラフルな熱気球が白鳥の上を飛んでいる、熱気球、複雑、8kの非常に専門的に詳細な、hdr、cgsociety", "🇯🇵 Japanese", "Artistic"
        ], 
    [
            "아름다운 반사 산 호수에서 두 개의 검은 백조가 머리를 만지는 사진, 화려한 열기구가 백조, 열기구, 복잡하고 전문적으로 상세한 8k, hdr, cgsociety","🇰🇷 Korean", "Artistic", 
        ], 

]

def run_demo(): 
    block = gr.Blocks(css=".container { max-width: 1300px; margin: auto; }")
    with block as demo:
    # with gr.Tabs(css=".container { max-width: 1300px; margin: auto; }"): 
    #     with gr.TabItem("Text to Image Generation"):
        gr.HTML(read_content("header_.html"))
        with gr.Group():
            # with gr.Box():

            with gr.Row().style(mobile_collapse=False, equal_height=True):
                
                with gr.Column(scale=1, min_width=500,):

                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        
                        with gr.Column(scale=4, min_width=100):
                            language_input = gr.Dropdown( ["🇱🇷 English", "🇻🇳 Vietnamese", "🇹🇼 TraditionalChinese", "🇨🇳 SimplifiedChinese", "🇫🇷 French", 
                            "🇩🇪 German","🇲🇨 Indonesian","🇯🇵 Japanese ","🇰🇷 Korean","🇪🇸 Spanish", "🇹🇭 Thai", ], value="🇱🇷 English", label="🌎 Choosing Your Language: 🇱🇷,🇻🇳,🇹🇼,🇨🇳,🇫🇷,🇩🇪,🇯🇵, others", show_label=True)
                        with gr.Column(scale=4, min_width=100,):
                            model_id = gr.Dropdown( ["Artistic", "Animation", "Realistic_v1", "Realistic_v2"], value="Artistic", label="🤖 Diffusion models ", show_label=True)
                        with gr.Column(scale=4, min_width=100,):
                            pallet_color = gr.Checkbox( value=True, label="color pallettes", show_label=True)

                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        # with gr.Column(scale=4, min_width=300):
                        text = gr.Textbox(lines=2, label="Input text prompt", placeholder="Typing: (what you want to edit in your image)..", show_label=True,).style(
                            border=(True, False, True, True),
                            rounded=(True, False, False, True),
                            container=False,)
                    
                  
                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                           
                            with gr.Column(scale=1, min_width=50):
                                promp_gen = gr.Button("Suggesting Prompt")
                                # .style(
                                #     margin=False,  border=(True, False, True, True),
                                #     rounded=(True, True, True, True),)

                            with gr.Column(scale=1, min_width=500):
                                text_generator = gr.Textbox(lines=5, label="Your Suggested Prompt", show_label=True,)
                                #.style(
                                #     border=(True, False, True, True),
                                #     rounded=(True, False, False, True),
                                #     container=True,)

                    promp_gen.click(fn=prompt_magic, inputs=[text, language_input], outputs=[text_generator])
     

                    with gr.Accordion("Advanced options", open=False):
                        
                        with gr.Row().style(mobile_collapse=False, equal_height=True):
                            
                            with gr.Column(scale=4, min_width=300, ):
                                seed = gr.Slider(
                                        label="Generating Different Image",
                                        minimum=0,
                                        maximum=2147483647,
                                        step=100,
                                        randomize=True,)
                            with gr.Column(scale=4, min_width=300, min_height=600):
                                generate_step = gr.Slider(label="Iteration Steps", minimum=1,
                                       maximum=75, value=25, step=1)    
                            #with gr.Row().style(mobile_collapse=False, equal_height=True):
                            with gr.Column(scale=4, min_width=300, min_height=600):
                                samples_num = gr.Slider(label="Number of Image",minimum=1, maximum=10, value=4, step=1,)  # show_label=False

                # option = gr.Radio(label=" Selecting Inpainting Area", default="Mask Area", choices=[
                #     "Mask Area", "Background Area"], show_label=True)
                with gr.Column(scale=1, min_width=500,):
                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        with gr.Column(scale=1,min_width=250,):# min_width=600, min_height=300
                                btn = gr.Button("Creating Images").style()# margin=False, rounded=(True, True, True, True),
                        with gr.Column(scale=1,min_width=250,):# min_width=400, min_height=300
                                stop = gr.Button(value="STOP")#.style(margin=False, rounded=(True, True, True, True),)

                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        #with gr.Column():  #scale=1, min_width=80, min_height=300
                        gallery = gr.Gallery(label="Edited images",show_label=True).style(grid=[2], height="auto")
                        #image_out = gr.Image(label="Edited Image", elem_id="output-img").style(height=400)
                        # with gr.Group(elem_id="share-btn-container"):
                            # community_icon = gr.HTML(community_icon_html, visible=False)
                            # loading_icon = gr.HTML(loading_icon_html, visible=False)
                            # share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)


            gr.Markdown("</center></h1> Prompts examples 📜 --> 🖼️. Models & Information Detail</center></h1>")

            
            ex = gr.Examples(examples=examples, fn=infer, inputs=[text, language_input, model_id,pallet_color, seed, generate_step ], outputs=[gallery], cache_examples=False, postprocess=False)
            text.submit(infer, inputs=[text,language_input, model_id, pallet_color,  samples_num,seed, generate_step], outputs=[gallery], postprocess=False)
            run_event=btn.click(fn=infer, inputs=[text, language_input, model_id,pallet_color,  samples_num,seed, generate_step], outputs=[gallery])
            stop.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

           
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
                            App Developer: @TranNhiem 🙋‍♂️ : 
                        </p>
                        </p>
                        <p style="align-items: center; margin-bottom: 7px;" >
                        <a This app power by (Natural Language Translation) Text-2-Image Diffusion Generative Model (StableDiffusion).</a>
                        </p>
                        </div>
                    </div>
                 
                """
            )

            #        gr.HTML(
            #     """
            #     <div class="footer">
            #         <p style="align-items: center; margin-bottom: 7px;" >

            #         </p>
            #         <div style="text-align: Center; font-size: 1.5em; font-weight: bold; margin-bottom: 0.5em;">
            #             <div style="
            #                 display: inline-flex; 
            #                 gap: 0.6rem; 
            #                 font-size: 1.0rem;
            #                 justify-content: center;
            #                 margin-bottom: 10px;
            #                 ">
            #             <p style="align-items: center; margin-bottom: 7px;" >
            #                 App Developer: @TranNhiem 🙋‍♂️ Connect with me on : 
            #             <a href="https://www.linkedin.com/feed/" style="text-decoration: underline;" target="_blank"> 🙌 Linkedin</a> ;  
            #                 <a href="https://twitter.com/TranRick2" style="text-decoration: underline;" target="_blank"> 🙌 Twitter</a> ; 
            #                 <a href="https://www.facebook.com/jean.tran.336" style="text-decoration: underline;" target="_blank"> 🙌 Facebook</a> 
            #             </p>
            #             </p>
            #             <p style="align-items: center; margin-bottom: 7px;" >
            #             <a This app power by (Natural Language Translation) Text-2-Image Diffusion Generative Model (StableDiffusion).</a>
            #             </p>
            #             </div>
            #         </div>
            #         <div style="
            #             display: inline-flex; 
            #             gap: 0.6rem; 
            #             font-size: 1.0rem;
            #             justify-content: center;
            #             margin-bottom: 8px;
            #             ">
            #             </p> 
                        
            #             <p>
            #             1. Natural Language Translation Model power by NLLB-200
            #             <a href="https://ai.facebook.com/research/no-language-left-behind/" style="text-decoration: underline;" target="_blank">NLLB</a>  
            #             </p>
            #             <p>
            #             2. Text-to-Image generative model power by Stable Diffusion 
            #             <a href="https://huggingface.co/CompVis" style="text-decoration: underline;" target="_blank">CompVis</a> and 
            #             <a href="https://huggingface.co/stabilityai" style="text-decoration: underline;" target="_blank">Stability AI</a> 
            #             </p>
            #         </div>

            #     </div>
            #     <div class="acknowledgments">
            #         <p><h4>LICENSE</h4>
            #         The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> 
            #         license. The authors claim no rights on the outputs you generate. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a>
            #         </p>
            #     </div>
            #     """
            # )
    
    demo.launch(share=True, enable_queue=True)  #server_name="172.17.0.1", # server_port=2222, share=True, enable_queue=True,  debug=True

# command to fix permission denied error
# sudo chmod -R 777 /home/jean/anaconda3/envs/jean/lib/python3.8/site-packages/gradio


if __name__ == '__main__':

    gr.close_all()
    run_demo()