# @TranNhiem 2022 12/05
'''
Features Design include:
1.. Creating MASK using brush to draw mask or creating mask..  
2.. Text to Image Inpainting (Mask area or Background area)
3.. Supporting multi-Language input  
4.. Creating Plugin Upsampling image for Generating, and Pallet Color
5.. Adding Painting by image (Image Painting)

## Installations requirements 
!pip install git+https://github.com/TranNhiem/diffusers.git
!pip install git+https://github.com/TranNhiem/CodeFormer.git

## Reference for Image Generation checkpoint model
Update Reference for Image Inpainting with Diffusion Models
https://huggingface.co/runwayml/stable-diffusion-inpainting 
https://huggingface.co/spaces/runwayml/stable-diffusion-inpainting/tree/main 

Note 2: Hardware Efficient with Xformer installation 
1.. Installation guide
2.. Implement it in your code

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
from diffusers import StableDiffusionInpaintPipeline
from glob import glob
from diffusers import LMSDiscreteScheduler, DDIMScheduler , EulerDiscreteScheduler, DPMSolverMultistepScheduler, DiffusionPipeline


## Library for Stable Super Resolution model 
from diffusers import StableDiffusionUpscalePipeline, LDMSuperResolutionPipeline
from torchvision import transforms

## API for Language Translation Model 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

## Library for Image Color Pallet
from colorthief import ColorThief
from pathlib import Path
import cv2
import math

## Upsacle and Restore Image with 
from codeformer_infer import inference

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

SD_weight_path="/data1/pretrained_weight/StableDiffusion/"

def dummy(images, **kwargs): return images, False

def read_content(file_path) :
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

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

def infer(prompt_, img,language_input, seed, option="Mask Area",samples_num=4, model_id="beta_1", upscale="upscale", pallet_color=True,  scale=7.5,steps_num=25,schedule_type='DPM_Solver', resize_image_method="BILINEAR", ): #option
    ## Checking the Image Size then Resize image
    sampling_resize= {
        "BILINEAR": PIL.Image.BILINEAR,
        "BICUBIC": PIL.Image.BICUBIC,
        "LANCZOS": PIL.Image.LANCZOS,
        "NEAREST": PIL.Image.NEAREST,
        "ANTIALIAS": PIL.Image.ANTIALIAS,
        "": PIL.Image.BICUBIC,
    }
    w, h = img["image"].size
    if w > 512:
        h = int(h * (512/w))
        w = 512
    if h > 512:
        w = int(w*(512/h))
        h = 512
    w, h = map(lambda x: x - x % 64, (w, h))
    # w //= 8
    # h //=8
    # w, h= 512, 512
    print("Your image Height and Width: ", h, w)
    mask = (img['mask'])
    mask = mask.resize((w, h), resample=sampling_resize[resize_image_method]) # [BILINEAR,BICUBIC, ANTIALIAS, LANCZOS, NEAREST]
  
    image=(img['image']).resize((w,h), resample=sampling_resize[resize_image_method])
    
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

    ## Language Translation
    #samples_num=int(samples_num/2)
    NLLB_path= "/data1/pretrained_weight/NLLB"
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B",cache_dir=NLLB_path )#cache_dir=NLLB_path
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B", cache_dir=NLLB_path)

    if source_langage[language_input] != "🇱🇷 English":
        print(f"source input language: {source_langage[language_input]}")
        translator_prompt = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_langage[language_input], tgt_lang='eng_Latn', max_length = 400)
        prompt= translator_prompt(prompt_)[0]
        prompt_=prompt['translation_text']
        print("Your English prompt translate from : ", prompt_)
        prompt_= [prompt_]*int(samples_num)
        del translator_prompt
        torch.cuda.empty_cache()
    
    else:
        prompt_= [prompt_]*int(samples_num)
        print(prompt_)
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    
    
    ###--------------------------------
    ## Stable Diffusion Model Section 
    ###--------------------------------
    ## beta_1 & beta_2 are better 
    model_id_={
        "beta_1": "runwayml/stable-diffusion-inpainting", 
        "beta_2": "stabilityai/stable-diffusion-2-inpainting", 
        # "beta_3": "prompthero/openjourney",
        # "beta_4": "CompVis/stable-diffusion-v1-4",
        # "Model-3": "runwayml/stable-diffusion-v1-5",
        #"Model-4": "stabilityai/stable-diffusion-2", 
    }
  
    generator = torch.Generator(device="cuda").manual_seed(int(seed)) # change the seed to get different results
        
    #LMSD=LMSDiscreteScheduler.from_config(model_id_["beta_1"], subfolder="scheduler")
    # LMSD = LMSDiscreteScheduler.from_config(model_id_["beta_1"], subfolder="scheduler")
    DDIMS = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    # Euler_1 = EulerDiscreteScheduler.from_pretrained(model_id_["beta_1"], subfolder="scheduler", prediction_type="v_prediction")
    Euler = EulerDiscreteScheduler.from_pretrained(model_id_[model_id], subfolder="scheduler", prediction_type="v_prediction")
    #pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        
    schedule_type_={
        "DPM_Solver": DPM_Solver,
        "DDIMS" : DDIMS, 
        "Euler": Euler
    }
    
    if schedule_type !="LMSD":
        pipeimg_1 = StableDiffusionInpaintPipeline.from_pretrained(
            model_id_[model_id], #
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir=SD_weight_path, 
            scheduler=schedule_type_[schedule_type],
            use_auth_token='token_value',
        ).to("cuda")

    else: 
        pipeimg_1 = StableDiffusionInpaintPipeline.from_pretrained(
            model_id_[model_id], #
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir=SD_weight_path, 
            use_auth_token='token_value',
        ).to("cuda")

        pipeimg_1.scheduler=LMSDiscreteScheduler.from_config(pipeimg_1.scheduler.config)


    pipeimg_1.safety_checker = dummy

    # Generate image for the masking area with prompt
    #with autocast("cuda"):#"cuda"
    #with torch.autocast("cuda"), torch.inference_mode():
    with torch.cuda.amp.autocast(dtype=torch.float16):
        images_1 = pipeimg_1(prompt=prompt_,image= image, mask_image=mask, paint_area=option,  height=h, width=w,
                                num_inference_steps=steps_num, guidance_scale=scale, generator=generator).images #negative_prompt=str(negative_prompt_)
    del pipeimg_1    
    torch.cuda.empty_cache()

    if upscale=="Restore":
        images=[]
        for id, img in enumerate(images_1): 
            img.save(f"/home/harry/BLIRL/SSL_Applications/inpainting_sd/img{id}.png")
        for i in range(len(images_1)): 
            img=f"/home/harry/BLIRL/SSL_Applications/inpainting_sd/img{i}.png"
            upscale_restore_img=inference(img, background_enhance= True, face_upsample= True, upscale= 1, codeformer_fidelity= 1.0, model_type="2x")
            images.append(upscale_restore_img)

    elif upscale=="Restore & Upscale":  
        images=[]
        for id, img in enumerate(images_1): 
            img.save(f"/home/harry/BLIRL/SSL_Applications/inpainting_sd/img{id}.png")
        for i in range(len(images_1)): 
            img=f"/home/harry/BLIRL/SSL_Applications/inpainting_sd/img{i}.png"
            upscale_restore_img=inference(img, background_enhance= True, face_upsample= True, upscale=4, codeformer_fidelity= 1.0, model_type="4x")
            images.append(upscale_restore_img)
    
    else: 
        images = images_1

    if pallet_color:
        print("*************Adding Pallet Color****************")
        w, h= images[0].size
        if upscale !="Restore & Upscale" or  upscale !="Restore":  
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

def paint_by_smaple(dict, reference, scale, seed, step, upscale, pallet_color, schedule_type, samples_num=4):

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
    

    
            
    #LMSD=LMSDiscreteScheduler.from_config(model_id_["beta_1"], subfolder="scheduler")
    # LMSD = LMSDiscreteScheduler.from_config(model_id_["beta_1"], subfolder="scheduler")
    DDIMS = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    # Euler_1 = EulerDiscreteScheduler.from_pretrained(model_id_["beta_1"], subfolder="scheduler", prediction_type="v_prediction")
    Euler = EulerDiscreteScheduler.from_pretrained("Fantasy-Studio/Paint-by-Example", subfolder="scheduler", prediction_type="v_prediction")
    #pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        
    ## Faster runing with scheduler 
    schedule_type_={
        "DPM_Solver": DPM_Solver,
        "DDIMS" : DDIMS, 
        "Euler": Euler
    }

    pipe = DiffusionPipeline.from_pretrained(
        "Fantasy-Studio/Paint-by-Example",
        torch_dtype=torch.float16,
        cache_dir=SD_weight_path, 
        scheduler=schedule_type_[schedule_type],
    )
    pipe = pipe.to("cuda")
    
    images_1 = pipe(
        image=init_image,
        mask_image=mask,
        example_image=reference,
        generator=generator,
        num_images_per_prompt=samples_num,
        guidance_scale=scale,
        num_inference_steps=step,
        
    ).images
    
    del pipe
    torch.cuda.empty_cache()

    if upscale =="Restore":
        images=[]
        for id, img in enumerate(images_1): 
            img.save(f"/home/harry/BLIRL/SSL_Applications/inpainting_sd/img{id}.png")
        for i in range(len(images_1)): 
            img=f"/home/harry/BLIRL/SSL_Applications/inpainting_sd/img{i}.png"
            upscale_restore_img=inference(img, background_enhance= True, face_upsample= True, upscale= 1, codeformer_fidelity= 1.0, model_type="2x")
            images.append(upscale_restore_img)

    elif upscale =="Restore & Upscale":  
        images=[]
        for id, img in enumerate(images_1): 
            img.save(f"/home/harry/BLIRL/SSL_Applications/inpainting_sd/img{id}.png")
        for i in range(len(images_1)): 
            img=f"/home/harry/BLIRL/SSL_Applications/inpainting_sd/img{i}.png"
            upscale_restore_img=inference(img, background_enhance= True, face_upsample= True, upscale=4, codeformer_fidelity= 1.0, model_type="4x")
            images.append(upscale_restore_img)
    
    else: 
        images = images_1

    if pallet_color:
        print("*************Adding Pallet Color****************")
        w, h= images[0].size
        if upscale !="Restore & Upscale" or  upscale !="Restore":  
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
    
    return images #, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def run_demo(): 
    
    block = gr.Blocks(css=".container { max-width: 1300px; margin: auto; }")
    with block as demo:
        gr.HTML(read_content("header.html"))
        with gr.Tabs():
           
            with gr.TabItem("🎨 Painting with Text 📄:"):
                gr.HTML(read_content("header__.html"))
                with gr.Group():
                
                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        with gr.Column(scale=4, min_width=200, min_height=600):
                            language_input = gr.Dropdown( ["🇱🇷 English", "🇻🇳 Vietnamese", "🇹🇼 TraditionalChinese", "🇨🇳 SimplifiedChinese", "🇫🇷 French", 
                            "🇩🇪 German","🇲🇨 Indonesian","🇯🇵 Japanese ","🇰🇷 Korean","🇪🇸 Spanish", "🇹🇭 Thai", ],value="🇱🇷 English", label="🌎 Choosing Your Language: 🇱🇷,🇻🇳,🇹🇼,🇨🇳,🇫🇷,🇩🇪,🇯🇵 ", show_label=True)
                        
                        with gr.Column(scale=4, min_width=700, min_height=600):
                            text = gr.Textbox(label="Your text prompt", placeholder="Typing: (what you want to edit in your image)..", show_label=True,lines=2, max_lines=3).style(
                                border=(True, False, True, True),
                                rounded=(True, False, False, True),
                                container=False,)

                    
                        #     negative_prompt = gr.Textbox(label="remove prompt", placeholder="Typing: (DON't want to have in generated image). Default is empty.", show_label=True, max_lines=1).style(
                        #         border=(True, False, True, True),
                        #         rounded=(True, False, False, True),
                        #         container=False,)

                        with gr.Column(scale=4, min_width=100, min_height=300):
                            with gr.Row().style(mobile_collapse=False, equal_height=True):
                                btn = gr.Button("Paint!")#.style(margin=False, rounded=(True, True, True, True),)
                            with gr.Row().style(mobile_collapse=False, equal_height=True):
                                stop_run = gr.Button("STOP")#.style(margin=False, rounded=(True, True, True, True),)

                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                            with gr.Column(scale=4, min_width=300, min_height=600):
                                #option = gr.Dropdown( ["Mask Area", "Background Area"], Value="Mask Area", label="Replacing Area", show_label=True)
                                option = gr.Radio(label="Inpainting Area", value="Mask Area", choices=["Mask Area", "Background Area"], show_label=True)

                            with gr.Column(scale=4, min_width=150, min_height=300):
                                model_id = gr.Dropdown( ["beta_1", "beta_2"], value="beta_1", label="🤖 Inpainting models ", show_label=True)
                            #resize_image_method = gr.Dropdown( ["BILINEAR","BICUBIC", "ANTIALIAS", "LANCZOS", "NEAREST"],label="Method Resize Image", show_label=True)
                            with gr.Column(scale=4, min_width=250, min_height=600):
                                upscale = gr.Dropdown( ["Original","Restore", "Restore & Upscale"], label="Upscale & Enhancement Image", value="Restore",)  # show_label=False
                            
                            with gr.Column(scale=4, min_width=200, min_height=600):
                                
                                with gr.Row().style(mobile_collapse=False, equal_height=True):
                                    samples_num =gr.Slider(label="Number of Image", minimum=1, maximum=4, value=2, step=1,)  # show_label=False
                                with gr.Row().style(mobile_collapse=False, equal_height=True):
                                    pallet_color = gr.Checkbox( value=True, label="color pallettes", show_label=True)

                            with gr.Column(scale=4, min_width=100, min_height=600):
                                # steps_num = gr.Slider(label="Generatio of steps", minimum=10, maximum=200, value=50, step=5,)  # show_label=False
                                seed= gr.Number(value=12032, label="Fixed Randomness", show_label=True)
                                # scale = gr.Slider(label="Guidance scale", minimum=0.0,
                                #                 maximum=30, value=7.5, step=0.1,)  # show_label=False
                
                            # with gr.Column(scale=4, min_width=100, min_height=300):
                            #     stop_run = gr.Button("STOP Run")#.style(margin=False, rounded=(True, True, True, True),)


                    with gr.Row().style(mobile_collapse=False,):#gallery

                        with gr.Column():#scale=1, min_width=100, min_height=400
                            with gr.Row():
                                with gr.Group():
                                    image = gr.Image(source="upload", tool="sketch",label="Input image",elem_id="image_upload", type="pil").style(height=400)
                                    #input_image_properties = gr.Textbox(label="Image Properties", max_lines=1)
                            with gr.Row():

                                with gr.Accordion("Advanced options", open=False):
                                    
                                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                                        with gr.Column(scale=4, min_width=300, min_height=600):
                                            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1,
                                                maximum=20, value=5.0, step=1)   
                                        with gr.Column(scale=4, min_width=300, min_height=600):
                                            generate_step = gr.Slider(label="Iteration Steps", minimum=20,
                                                maximum=120, value=30, step=5)    
                                        #with gr.Row().style(mobile_collapse=False, equal_height=True):
                                        with gr.Column(scale=4, min_width=300, min_height=600):
                                            scheduler_type =  gr.Dropdown(["LMSD","Euler","DPM_Solver", "DDIMS"], label="Sampling method", value="DPM_Solver",) # show_label=False

                        with gr.Column():  #scale=1, min_width=80, min_height=300
                            gallery = gr.Gallery(label="Edited images",show_label=True).style(grid=[2], height="auto").style(height=400)
                    


                    run_event=btn.click(fn=infer, inputs=[text,image,language_input,seed, option, samples_num, model_id, upscale, pallet_color,guidance_scale, generate_step, scheduler_type ], outputs=[gallery ])
                    stop_run.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

            with gr.TabItem("🎨 Painting with Image 🖼️:"):
                gr.HTML(read_content("header_.html"))
                with gr.Group():
                    with gr.Box():
                        with gr.Row(mobile_collapse=False, equal_height=True):
                            with gr.Column(equal_height=True):
                                image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Source Image")
                                reference = gr.Image(source='upload', elem_id="image_upload", type="pil", label="Reference Image")

                            with gr.Column(equal_height=True):
                                with gr.Row().style(mobile_collapse=False, equal_height=True):
                                    with gr.Column(scale=4, min_width=150):
                                        samples_num =gr.Slider(label="Number of Image", minimum=1, maximum=4, value=2, step=1,)  # show_label=False
                                    with gr.Column(scale=4, min_width=100):
                                        pallet_color = gr.Checkbox( value=True, label="color pallettes", show_label=True)
                                    with gr.Column(scale=4, min_width=250):
                                            upscale = gr.Dropdown( ["Original","Restore", "Restore & Upscale"], label="Upscale & Enhancement Image", value="Restore",)  # show_label=False
                    
                                with gr.Row().style(mobile_collapse=False, equal_height=True):
                                    with gr.Column(scale=4, min_width=300):
                                        btn = gr.Button("Paint!").style(
                                            margin=False,
                                            rounded=(False, True, True, False),
                                            full_width=True,
                                        )
                                    with gr.Column(scale=4, min_width=200):
                                        stop_run = gr.Button("STOP")#.style(margin=False, rounded=(True, True, True, True),)


                                #image_out = gr.Image(label="Editted Image", elem_id="output-img").style(height=400)
                                image_out = gr.Gallery(label="Edited images",show_label=True).style(grid=[2], height="auto").style(height=400)
                                
                                # with gr.Row():
                                with gr.Accordion("Advanced options", open=False):
                                    
                                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                                        with gr.Column(scale=4, min_width=200, min_height=600):
                                            guidance = gr.Slider(label="Guidance Scale", minimum=1,
                                                maximum=20, value=5.0, step=1)   
                                        with gr.Column(scale=4, min_width=200, min_height=600):
                                            steps = gr.Slider(label="Iteration Steps", minimum=20,
                                                maximum=120, value=30, step=5)    
                                        #with gr.Row().style(mobile_collapse=False, equal_height=True):
                                        with gr.Column(scale=4, min_width=200, min_height=600):
                                            scheduler_type =  gr.Dropdown(["Euler","DPM_Solver", "DDIMS"], label="Sampling method", value="DPM_Solver",) # show_label=False
                                        with gr.Column(scale=4, min_width=200, min_height=600):
                                            seed = gr.Slider(0, 10000, label='Fixed Randomness', value=0, step=1)
                                        
                        
                        run_event=btn.click(fn=paint_by_smaple, inputs=[image, reference, guidance, seed, steps, upscale, pallet_color,scheduler_type,samples_num ], outputs=[image_out])
                        stop_run.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

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
                        </p>
                        <p style="align-items: center; margin-bottom: 7px;" >
                        <a This app power by (Natural Language Translation) Text-2-Image Diffusion Generative Model (StableDiffusion).</a>
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
                        
                        <p>
                        1. Natural Language Translation Model power by NLLB-200
                        <a href="https://ai.facebook.com/research/no-language-left-behind/" style="text-decoration: underline;" target="_blank">NLLB</a>  
                        </p>
                        <p>
                        2. Text-to-Image generative model power by Stable Diffusion 
                        <a href="https://huggingface.co/CompVis" style="text-decoration: underline;" target="_blank">CompVis</a> and 
                        <a href="https://huggingface.co/stabilityai" style="text-decoration: underline;" target="_blank">Stability AI</a> 
                        3. Paint by example
                        <a href="https://huggingface.co/Fantasy-Studio/Paint-by-Example" style="text-decoration: underline;" target="_blank">Fantasy Studio</a> 
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
        demo.launch( share=True, enable_queue=True,  debug=True)  #server_name="172.17.0.1", # server_port=2222, share=True, enable_queue=True,  debug=True

if __name__ == '__main__':

    gr.close_all()
    run_demo()