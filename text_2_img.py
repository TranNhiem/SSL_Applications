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
import numpy as np 
from min_dalle import MinDalle 
from torchvision import transforms
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Loading Dalle-E mini or Mega  model
model= MinDalle(
    models_root= "/home/rick/pretrained_weight/Dalle_mini_mega/pretrained", #/home/rick/pretrained_weights/Dalle_mini_mega
    dtype= torch.float16, 
    device='cuda', 
    is_mega= True, # False -> Using mini model
)
def init_img_dalle(prompt): 
    ## Using Dalle-E mini or Mega to generate the initial image
    image= model.generate_images( text= prompt,
                                    seed=-1,
                                    grid_size=1,
                                    is_seamless=False,
                                    temperature=1,
                                    top_k=256,
                                    supercondition_factor=16,
                                    is_verbose=False,
                                    )
    ## Transform the tensor image to PIL image
    print(image.shape)
    image= transforms.ToPILImage()(image[0].permute(2,0,1))
    return image

image = init_img_dalle("A bird with a long beak")
ext = 'jpg' 
filename = './Bird_images/dalle_bird_image'
image_path = '{}.{}'.format(filename, ext)
image.save(image_path)