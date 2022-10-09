# TranNhiem 2022/09
"""
Text to image generation with both different approaches 
1. Dalle-E mini + Mega, # https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-Mini-Explained-with-Demo--Vmlldzo4NjIxODA 
  + official repository: https://github.com/borisdayma/dalle-mini 
  + Experiment training:  https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-Mega-Training-Journal--VmlldzoxODMxMDI2

2. Stable Diffusion 

Text to image generation with combination of two approaches 
Text --> Dalle-E --> Initial Image --> Stable Diffusion Enhance the image with addition Prompt. 

"""
from email.errors import FirstHeaderLineIsContinuationDefect
from min_dalle import MinDalle
import numpy as np
from PIL import Image
import torch
import gradio as gr
from stable_diffusion_model import StableDiffusionInpaintingPipeline_, StableDiffusionPipeline, StableDiffusionCLIP_Guided, StableDiffusionPipelineAIT
from torchvision import transforms
import PIL
from pathlib import Path
import os
from diffusers import LMSDiscreteScheduler
from transformers import CLIPFeatureExtractor, CLIPModel
import random 
import time 
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# Using original stable diffusion pipeline

# ---------------------------------------------------------
# Loading Dalle-E Image Generation
# ---------------------------------------------------------
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

#prompt = "A little boy wearing headphones and looking at computer monitor"
generator = torch.Generator(device='cuda').manual_seed(20000)

def dalle_text_2_image(prompt, dalle_model="Dalle-mega", stream_image=False):
    model = MinDalle(
        # /home/rick/pretrained_weights/Dalle_mini_mega
        models_root="/data/pretrained_weight/Dalle_mini_mega",
        dtype=torch.float32,
        device='cuda',
        is_mega=False if dalle_model == "Dalle-mini" else True,  # False -> Using mini model,
        is_reusable=True,
    )

    def init_img_dalle(prompt, img_size=512, stream_image=True, save_sub_img=False):
        # Using Dalle-E mini or Mega to generate the initial image
        if stream_image:
            with torch.autocast("cuda"):
                images_gen = model.generate_image_stream(text=prompt,
                                                         seed=-1,
                                                         grid_size=1,
                                                         is_seamless=True,
                                                         temperature=1.5,
                                                         progressive_outputs=True,
                                                         top_k=img_size,
                                                         supercondition_factor=16.,
                                                         # is_verbose=False
                                                         )

            for id, img in enumerate(images_gen):
                if save_sub_img:
                    img.save("./Bird_images/" +
                             prompt.replace(" ", "_")+str(id)+".jpg")
                else:
                    continue

                   #img.save("./Bird_images/"+prompt.replace(" ", "_")+".jpg")
            return img

        else:
            with torch.autocast("cuda"):
                image = model.generate_image(text=prompt,
                                             seed=-1,
                                             grid_size=1,
                                             is_seamless=False,  # If this set to False --> Return tensor
                                             temperature=2,
                                             top_k=img_size,
                                             supercondition_factor=16.,
                                             is_verbose=False
                                             )
                # Transform the tensor image to PIL image
                # print(f"checking image shape : {image.shape}")
                # image= transforms.ToPILImage()(image[0].permute(2,0,1))
                #image.save("./Bird_images/"+prompt.replace(" ", "_")+".jpg")
                return image

    image = init_img_dalle(prompt, img_size=128, stream_image=stream_image,)
    #print(f"checking image shape : {image.size}")

    torch.cuda.empty_cache()
    return image
#image= dalle_text_2_image(prompt)
# ---------------------------------------------------------
# Section 1  DALLE to Stable Diffusion Model
# ---------------------------------------------------------
def dalle_to_SD(prompt, image_width, image_height, samples_num, step_num, scale, option,):

    pipeimg = StableDiffusionInpaintingPipeline_.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
    ).to("cuda")

    ## Preprocessing image
    def preprocess(image):
        w, h = image.size
        if w > 512:
            h = int(h * (512/w))
            w = 512
        if h > 512:
            w = int(w*(512/h))
            h = 512
        # resize to integer multiple of 32
        w, h = map(lambda x: x - x % 64, (w, h))
        #w, h = map(lambda x: x - x % 32, (w, h))
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.*image - 1.

    def dummy(images, **kwargs): return images, False

    pipeimg.safety_checker = dummy

    # This section Get the image form Dalle MODEL

    init_img = dalle_text_2_image(
        prompt, dalle_model=option, stream_image=True)

    # init_img= Image.open("/code_spec/SSL_Application/SSL_Applications/Bird_images/"+prompt.replace(" ", "_")+".jpg")#.convert("RGB")

    init_img_ = init_img.resize((image_width, image_height))
    init_img = preprocess(init_img_)
    with torch.autocast("cuda"):
        images = pipeimg(prompt=[prompt]*(samples_num-1),
                         init_image=init_img,
                         # strength= strength,
                         num_inference_steps=step_num,
                         guidance_scale=scale,
                         # strength=strength,
                         generator=generator,
                         inpainting_v2=False,
                         )["sample"]

    #images[0].save("./Bird_images/"+prompt.replace(" ", "_")+"SD_Variant"+".jpg")
    return [init_img_], images
# img=dalle_to_SD(prompt)

# ---------------------------------------------------------
# Section 2 Stable Diffusion Model text_2_image
# ---------------------------------------------------------
def text_2_image(prompt, path):
    pipeimg = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
    ).to("cuda")
    samples_num = 2
    init_img = None
    mode = "prompt"
    num_inference_steps = 80
    guidance_scale = 7.5
    # with torch.cuda.amp.autocast():
    start_time = time.time()
    with torch.autocast("cuda"):
        outputs = pipeimg(prompt=[prompt]*samples_num,
                          mode=mode,
                          height=512,
                          width=512,
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale,
                          # init_image=init_img,
                          generator=generator,
                          strength=0.8,
                          return_intermediates=False,
                          )

    images = outputs["sample"]
    #time_inference = outputs["time"]
    #print("The inference time is : ", time_inference)
    images[0].save(path  + "test_AiT"+ str(i) + ".jpg")
  
    print("--- Normal SD pipeline %s seconds ---" % (time.time() - start_time))
   
    return images


def text_2_image_AiT_inference(prompt, path):
    img_pipe= StableDiffusionPipelineAIT.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,).to("cuda")
    samples_num = 2
    init_img = None
    
    num_inference_steps = 80
    guidance_scale = 7.5
    # with torch.cuda.amp.autocast():
    start_time = time.time()
    with torch.autocast("cuda"):
        images = img_pipe(prompt=[prompt]*samples_num,
                          height=512,
                          width=512,
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale,
                          # init_image=init_img,
                          generator=generator,
                          strength=0.8,
                          return_intermediates=False,
                          ).images[0]

        images.save(path  + "test_AiT"+ str(i) + ".jpg")
    
    print("--- AiT SD pipeline %s seconds ---" % (time.time() - start_time))
    
    return images 
#image = text_2_image(prompt)
# text_2_image(prompt) Gradio App Demo
# ---------------------------------------------------------
# def text_2_image_gradio():
block = gr.Blocks(css=".container { max-width: 1300px; margin: auto; }")

examples = [['A busy city with buildings covered in paint in the style of moebius, james jean, painterly, yoshitaka amano, hiroshi yoshida, loish, painterly, and artgerm, illustration'],
            ['Movie still photo of cute anthropomorphic vulpes vulpes fulva as a cow girl in wild west running in a gunfight, rugged clothes, motion blur, bullets whizzing by weta, greg rutkowski, wlop, ilya kuvshinov, rossdraws, artgerm, octane render, iridescent, bright morning, anime, liosh, mucha'],
            ['Photo, a squirrel fighting a rabbit, woodland location, stefan kostic and david cronenberg, realistic, sharp focus, 8 k high definition, intricate, chiaroscuro, elegant, perfect faces, symmetrical face, extremely detailed, hypnotic eyes, realistic, fantasy art, masterpiece zdzislaw beksinski, national geographic, artgerm'],
            ['Greg manchess portrait painting of armored teenage mutant ninja turtles as overwatch character, medium shot, asymmetrical, profile picture, organic painting, sunny day, matte painting, bold shapes, hard edges, street art, trending on artstation, by huang guangjian and gil elvgren and sachin teng'],
            ]
def text_2_img_demo_luanch():
    with block as demo:

        gr.Markdown(
            "<h1><center> Text to Image 📜 ->> 🖼️ </center></h1>")
        gr.Markdown(
            "<h2><center> Text2Image Power by DALLE-mini/mega & StableDiffusion </center></h2>")

        with gr.Group():
            with gr.Box():

                with gr.Row().style(mobile_collapse=False, equal_height=True):
                    text = gr.Textbox(label="1st: 📜 Your prompt describes the expected Image Ouput", placeholder="Enter your prompt here...", show_label=True, max_lines=1).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True), container=False,)

                with gr.Row().style(mobile_collapse=False, equal_height=True):
                    gr.Markdown(
                        "<h3><center> 2nd: Setting Parameters control generate image output </center></h3>")

                with gr.Row().style(mobile_collapse=False, equal_height=True):

                    with gr.Column(scale=1, min_width=600):
                        image_height = gr.Slider(
                            label="Image Height", minimum=256, maximum=1024, step=50, value=512, show_label=True)
                        image_width = gr.Slider(
                            label="Image Width", minimum=256, maximum=1024, step=50, value=512, show_label=True)
                        number_image = gr.Slider(
                            label="Number of Images", minimum=2, maximum=10, value=2, step=1, show_label=True)

                    with gr.Column(scale=2, min_width=400):
                        gr.Markdown(
                            "<h4><center> StableDiffusion Parameters control generate image output </center></h4>")
                        steps_num = gr.Slider(
                            label="Generatio of steps", minimum=40, maximum=200, value=80, step=5,)  # show_label=False
                        # show_label=False
                        scale = gr.Slider(
                            label="Guidance scale", minimum=0.0, maximum=30, value=7.5, step=1,)
                        # strength = gr.Slider(
                        #     label="Strength", minimum=0.0, maximum=1.0, value=0.8, step=0.05,)  # show_label=False

                    with gr.Column(scale=2, min_width=50):

                        btn = gr.Button("4th: Run").style(
                            margin=False, rounded=(True, True, True, True),)
                        option = gr.Radio(label="3rd:  DALLE Model", choices=[
                            "Dalle-min", "Dalle-mega", ], show_label=True)

            with gr.Row().style(mobile_collapse=False, equal_height=True):  # equal_height=True

                with gr.Column(scale=1, min_width=400):
                    Dalle_image = gr.Gallery(
                        label='DALLE Image').style(grid=[1], height="auto",)  # container=True

                with gr.Column(scale=2, min_width=600, ):  # min_height=400
                    SD_image = gr.Gallery(label='StableDiffusion Variant Image').style(
                        grid=[2, 2], height="auto",)  # container=True
                    #image_box = gr.Image(label='Expected Replacement Object')

            gr.Markdown("## Examples text 📜 prompt.")

            # if option is not None:
            ex = gr.Examples(examples=examples, fn=dalle_to_SD, inputs=[
                text, image_width, image_height, number_image, steps_num, scale, option], outputs=[Dalle_image, SD_image], cache_examples=False)

            #ex.dataset.headers = [""]
            text.submit(fn=dalle_to_SD, inputs=[
                text, image_width, image_height, number_image, steps_num, scale, option], outputs=[Dalle_image, SD_image], )

            btn.click(fn=dalle_to_SD, inputs=[
                text, image_width, image_height, number_image, steps_num, scale, option], outputs=[Dalle_image, SD_image], )

            # else:
            #     print("Adding new features for Not using Dalle Model ")

    # server_name="13.65.38.227", server_port=1111,# enable_queue=True, # show_error=True, debug=True,
    demo.launch(server_name="127.0.0.1", server_port=2222, share=True,
                enable_queue=True, )  # show_error=True, debug=True,


##--------------------------------------------------------------------
## Section 3 Stable Diffusion Guided by CLIP image feature embedding 
##--------------------------------------------------------------------

## Choosing the pretrained weight for the CLIP model can be found here 
# https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/pretrained.py
# 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K/' ## Best accuracy zero-shot 78.0%
# 'laion/CLIP-ViT-g-14-laion2B-s12B-b42K/'
clip_model_id= "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
scheduler= "plms" # ["plms", "lms"] 

def SD_Clip_pipeline(SD_model_id= "CompVis/stable-diffusion-v1-4",clip_id=clip_model_id,scheduler=scheduler  ): 
    SD_pipeline= StableDiffusionPipeline.from_pretrained(
       SD_model_id, revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
    )#.to("cuda")
    if scheduler == "lms": 
        scheduler= LMSDiscreteScheduler(beta_start= 0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    else: 
        scheduler= SD_pipeline.scheduler 
    clip_model= CLIPModel.from_pretrained(clip_id, torch_dtype=torch.float16)#.to("cuda")
    feature_extractor= CLIPFeatureExtractor.from_pretrained(clip_id, torch_dtype=torch.float16)#.to("cuda")
    
    sd_clip_guided_pipeline= StableDiffusionCLIP_Guided(
        unet= SD_pipeline.unet,
        vae= SD_pipeline.vae,
        tokenizer= SD_pipeline.tokenizer,
        text_encoder=SD_pipeline.text_encoder,
        scheduler= scheduler,
        clip_model= clip_model,
        feature_extractor= feature_extractor,
    )
    return sd_clip_guided_pipeline

def SD_Clip_guied(prompt, num_samples=4, clip_prompt="", num_inference_steps=50, guidance_scale=7.5, clip_guidance_scale=100,
                    num_cutouts = 4, use_cutouts: bool= False, unfreeze_unet: bool = False, unfreeze_vae: bool = False, seed: int = 3202020,save_path= None,): 
    sd_clip_guided_pipeline= SD_Clip_pipeline(clip_id=clip_model_id, scheduler=scheduler)
    sd_clip_guided_pipeline=sd_clip_guided_pipeline.to("cuda")

   
    if unfreeze_unet:
        sd_clip_guided_pipeline.unfreeze_unet()
    else:
        sd_clip_guided_pipeline.freeze_unet()
    
    if unfreeze_vae:
        sd_clip_guided_pipeline.unfreeze_vae()
    else:
        sd_clip_guided_pipeline.freeze_vae()

    generator = torch.Generator(device="cuda").manual_seed(random.randint(0, seed)) #seed
    
    images= [] 
    for i in range(num_samples):
        with torch.autocast("cuda"):
            image = sd_clip_guided_pipeline(
                prompt=prompt,
                clip_prompt= clip_prompt if clip_prompt.strip() != "" else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                clip_guidance_scale=clip_guidance_scale,
                num_cutouts=num_cutouts,
                use_cutouts=True if use_cutouts else False,
                generator=generator,
            ).images[0]
        images.append(image)
        if save_path is not None:
            Path(save_path+ "val2014/").mkdir(parents=True, exist_ok=True)
            Path(save_path+ "train2014/").mkdir(parents=True, exist_ok=True)
            print(image.size)
            image.save(save_path  + "test_rand_seed"+ str(i) + ".jpg")
    return images

# path_save="/data1/coco_synthetic_CLIP_Guided_SD/"
# prompt="A busy city with buildings covered in paint in the style of moebius, james jean, painterly, yoshitaka amano, hiroshi yoshida, loish, painterly, and artgerm, illustration"
# imgs= SD_Clip_guied(prompt= prompt, save_path=path_save )

###----------------- CLIP Guided Diffusion Continue -----------------###