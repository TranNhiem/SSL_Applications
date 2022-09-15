## TranNhiem 2022/09 
"""
Text to image generation with both different approaches 
1. Dalle-E mini + Mega, # https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-Mini-Explained-with-Demo--Vmlldzo4NjIxODA 
  + official repository: https://github.com/borisdayma/dalle-mini 
  + Experiment training:  https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-Mega-Training-Journal--VmlldzoxODMxMDI2

2. Stable Diffusion 

Text to image generation with combination of two approaches 
Text --> Dalle-E --> Initial Image --> Stable Diffusion Enhance the image with addition Prompt. 

"""
import os 
import torch 
from PIL import Image 
import PIL
import numpy as np 
from min_dalle import MinDalle 
from torchvision import transforms
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from stable_diffusion_model import StableDiffusionInpaintingPipeline_, StableDiffusion_text_image_to_image_

# ---------------------------------------------------------
# Loading Dalle-E Image Generation 
# ---------------------------------------------------------
prompt="An avocado landed on moon"
generator = torch.Generator(device='cuda').manual_seed(1024)
def dalle_text_2_image(prompt):
    model= MinDalle(
        models_root= "/home/rick/pretrained_weight/Dalle_mini_mega/pretrained", #/home/rick/pretrained_weights/Dalle_mini_mega
        dtype= torch.float32, 
        device='cuda', 
        is_mega= False, # False -> Using mini model, 
        is_reusable= True, 
    )
    def init_img_dalle(prompt, img_size= 512, stream_image=True,save_sub_img=False): 
        ## Using Dalle-E mini or Mega to generate the initial image
        if stream_image:
            images_gen= model.generate_image_stream( text= prompt,
                                            seed=-1,
                                            grid_size=1,
                                            is_seamless=True,
                                            temperature=2,
                                            progressive_outputs= True,
                                            top_k=img_size,
                                            supercondition_factor=16.,
                                            #is_verbose=False
                                            )
            
            for id, img in enumerate(images_gen): 
                if save_sub_img: 
                    img.save("./Bird_images/"+prompt.replace(" ", "_")+str(id)+".jpg")
                else: 
                    img.save("./Bird_images/"+prompt.replace(" ", "_")+".jpg")
            return img

        else: 
            image= model.generate_image( text= prompt,
                                        seed=-1,
                                        grid_size=1,
                                        is_seamless=False,# If this set to False --> Return tensor
                                        temperature=2,
                                        top_k=img_size,
                                        supercondition_factor=16.,
                                        is_verbose=False
                                        )
            ## Transform the tensor image to PIL image
            # print(f"checking image shape : {image.shape}")
            # image= transforms.ToPILImage()(image[0].permute(2,0,1))
            image.save("./Bird_images/"+prompt.replace(" ", "_")+"."+".jpg")
            return image
    
    image = init_img_dalle(prompt, img_size=256, stream_image=True,)
    print(f"checking image shape : {image.size}")

    torch.cuda.empty_cache()
    return image
# ---------------------------------------------------------
# Section DALLE to Stable Diffusion Model 
# ---------------------------------------------------------
def dalle_to_SD(prompt):

    pipeimg = StableDiffusionInpaintingPipeline_.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
    ).to("cuda")
    ## Preprocessing image 
    def preprocess(image):
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.*image - 1.

    def dummy(images, **kwargs): return images, False

    pipeimg.safety_checker = dummy
   
    samples_num=1
    step_num=50
    scale=7.5
    init_img= Image.open("./Bird_images/"+prompt.replace(" ", "_")+".jpg").convert("RGB")
    init_img = init_img.resize((768, 512))
    init_img= preprocess(init_img)


    with torch.cuda.amp.autocast():
        images = pipeimg(prompt=[prompt]*samples_num,
                            init_image=init_img,
                            # strength= strength,
                            num_inference_steps=step_num,
                            guidance_scale=scale,
                            generator=generator,
                            inpainting_v2=False,
                            )["sample"]
    images[0].save("./Bird_images/"+prompt.replace(" ", "_")+"SD_Variant"+".jpg")

    return images

# ---------------------------------------------------------
# Section Stable Diffusion Model text_2_image
# ---------------------------------------------------------
pipeimg = StableDiffusion_text_image_to_image_.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
    ).to("cuda")

samples_num=1
init_img=None
mode="prompt"

with torch.cuda.amp.autocast():
    images = pipeimg(prompt=[prompt]*samples_num,
                                mode=mode, 
                                using_clip= False, 
                                height = 512,
                                width = 512,
                                init_image=init_img,
                                generator = generator,
                                strength = 0.8,
                                return_intermediate = False,
                                )["sample"]
images[0].save("./Bird_images/"+prompt.replace(" ", "_")+"SD_text_2_image"+".jpg")
