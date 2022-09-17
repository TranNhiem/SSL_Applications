import os 
import re
import json 
import random
import torch 
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from stable_diffusion_model import StableDiffusionPipeline
from PIL import Image
import sys
from pathlib import Path
#CUDA_VISIBLE_DEVICES="1"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from min_dalle import MinDalle
#CUDA_VISIBLE_DEVICES=2,3 python xxx.py
# device = torch.device(
#     "cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f"Using available: {device}")

## Preprocessing the caption with some special characters
def pre_caption(caption, max_words=50): 
    caption= re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
    caption= re.sub(r"\s{2,}", ' ', caption)
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    ## Truncate caption  
    caption_words= caption.split(' ')
    caption= caption.strip(' ')

    ## Truncate cpation 
    caption_words= caption.split(' ')
    if len(caption_words) > max_words:
        caption_words= caption_words[:max_words]
        caption= ' '.join(caption_words)
    
    return caption 

## Generated data given by the text description 
class COCO_synthetic_Dataset(Dataset):

    def __init__(self, image_root, ann_root, max_words=200, prompt='', guidance_scale=7.5,  num_inference_steps=70, seed=123245):
        '''
        image_root (string): Root directory for storing the generated images (ex: /data/coco_synthetic/)
        anno_root(string): directory for storing the human caption file from COCO Caption dataset
        '''
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'
        Path(image_root+ "val2014/").mkdir(parents=True, exist_ok=True)
        Path(ann_root).mkdir(parents=True, exist_ok=True)
        download_url (url, ann_root)
        self.annotation= json.load(open(os.path.join(ann_root, filename), 'r'))

        self.image_root= image_root 
        self.max_words= max_words
        self.prompt= prompt
        self.guidance_scale= guidance_scale
        self.num_inference_steps= num_inference_steps
        self.generator = torch.Generator(device="cuda").manual_seed(seed) #random.randint(0, 100000) #random.randint(0,10000) # change the seed to get different results
        self.append_id=["test"]
        self.model= StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
    ).to("cuda")

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        ann= self.annotation[idx]
        
        caption= self.prompt + pre_caption(ann['caption'], self.max_words)
        image_id= ann['image_id']
        image_name= ann['image'] ## Saved image's name
        
        if image_id == self.append_id[-1]:
            print("Using mode Image to generate image")
            init_image= Image.open(os.path.join(self.image_root, image_name))
            with torch.autocast('cuda'):
                generate_image= self.model(
                                prompt=[caption],
                                mode="image",
                                height=512,
                                width=512,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                init_image=init_image,
                                generator=self.generator,
                                strength=0.8,
                                return_intermediates=False,
                                )['sample']
        else: ## Case not repeat image
            print("Using mode Prompt to generate image")
            with torch.autocast('cuda'):
                generate_image= self.model(
                                prompt=[caption],
                                mode="prompt",
                                height=512,
                                width=512,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                init_image=None,
                                generator=self.generator,
                                strength=0.8,
                                return_intermediates=False,
                                )['sample']

        generate_image[0].save(os.path.join(self.image_root, image_name))
        self.append_id.append(image_id)
        print(f"image name {caption} Generated")
        return image_name 

generate_data= COCO_synthetic_Dataset(image_root='/data1/coco_synthetic/', ann_root='/data1/coco_synthetic/')
print(generate_data.__len__())
# for i in range(5):
#     generate_data.__getitem__(i)
print("------------------------ Done ------------------------")
# for i in range(4, 6):
#     generate_data.__getitem__(i)


class COCO_synthetic_Dalle_SD(Dataset): 

    def __init__(self, image_root, ann_root, max_words=200, prompt='highly detailed', 
                    dalle_topk=128, temperature=1., supercondition_factor=16.,
                    guidance_scale=7.5,  num_inference_steps=70, seed=123245):
        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'
        Path(image_root+ "val2014/").mkdir(parents=True, exist_ok=True)
        Path(ann_root).mkdir(parents=True, exist_ok=True)
        download_url (url, ann_root)
        self.annotation= json.load(open(os.path.join(ann_root, filename), 'r'))

        self.image_root= image_root 
        self.max_words= max_words
        self.prompt= prompt
        ## Parameter for Dalle-Mini Model
        self.dalle_topk=dalle_topk
        self.temeperature= temperature 
        self.supercondition_factor= supercondition_factor 
        self.Dalle_model = MinDalle(
        # /home/rick/pretrained_weights/Dalle_mini_mega
            models_root="/data1/pretrained_weight/Dalle_mini_mega",
            dtype=torch.float32,
            device="cuda",
            is_mega=False,  # False -> Using mini model,
            is_reusable=True,)

        ## Parameter for StableDiffusion Model 
        self.guidance_scale= guidance_scale
        self.num_inference_steps= num_inference_steps
        self.generator = torch.Generator(device="cuda").manual_seed(seed) #random.randint(0, 100000) #random.randint(0,10000) # change the seed to get different results
        self.append_id=["test"]

        self.SD_model= StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,).to("cuda")
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        ann= self.annotation[idx]
        
        caption= self.prompt + pre_caption(ann['caption'], self.max_words)
        image_id= ann['image_id']
        image_name= ann['image'] ## Saved image's name
        
       
        ## Case not repeat image
        with torch.autocast('cuda'):
            init_image= self.Dalle_model.generate_image(text=caption,
                                             seed=-1,
                                             grid_size=1,
                                             is_seamless=False,  # If this set to False --> Return tensor
                                             temperature=self.temeperature,
                                             top_k=self.dalle_topk,
                                             supercondition_factor=self.supercondition_factor,
                                             is_verbose=False
                                             ) 
            init_image = init_image.resize((512, 512))
            generate_image= self.SD_model(
                            prompt=[caption],
                            mode="image",
                            height=512,
                            width=512,
                            num_inference_steps=self.num_inference_steps,
                            guidance_scale=self.guidance_scale,
                            init_image=init_image,
                            generator=self.generator,
                            strength=0.8,
                            return_intermediates=False,
                            )['sample']

        generate_image[0].save(os.path.join(self.image_root, image_name))
        self.append_id.append(image_id)
        print(f"image name {caption} Generated")
        return image_name 

generate_data_= COCO_synthetic_Dalle_SD(image_root='/data1/coco_synthetic/', ann_root='/data1/coco_synthetic/')

for i in range(5):
    generate_data_.__getitem__(i)