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

# Preprocessing the caption with some special characters


def pre_caption(caption, max_words=50):
    caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
    caption = re.sub(r"\s{2,}", ' ', caption)
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # Truncate caption
    caption_words = caption.split(' ')
    caption = caption.strip(' ')

    # Truncate cpation
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption_words = caption_words[:max_words]
        caption = ' '.join(caption_words)

    return caption

## Generated data given by the text description 
class COCO_synthetic_Dataset(Dataset):

    def __init__(self, image_root, ann_root, max_words=200, prompt='4k , highly detailed', generate_mode="repeat", 
                         guidance_scale=7.5,  num_inference_steps=70, seed=123245):
        '''
        image_root (string): Root directory for storing the generated images (ex: /data/coco_synthetic/)
        anno_root(string): directory for storing the human caption file from COCO Caption dataset
        '''
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'

        Path(image_root + "val2014/").mkdir(parents=True, exist_ok=True)
        Path(image_root + "train2014/").mkdir(parents=True, exist_ok=True)

        Path(ann_root).mkdir(parents=True, exist_ok=True)

        # os.makedirs(image_root+ "val2014/", exist_ok=True)
        # os.makedirs(ann_root, exist_ok=True)

        download_url(url, ann_root)
        self.annotation = json.load(
            open(os.path.join(ann_root, filename), 'r'))

        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generate_mode= generate_mode
        # random.randint(0, 100000) #random.randint(0,10000) # change the seed to get different results
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        self.append_id = ["test"]
        self.repeat_name=["test"]
        self.append_id_repeat=["test"]
        self.model = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="fp16",
            torch_dtype=torch.float32,
            use_auth_token=True,
        ).to("cuda")
        
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        ann = self.annotation[idx]

        caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        image_id = ann['image_id']
        image_name = ann['image']  # Saved image's name
        path=os.path.join(self.image_root, image_name)
        if self.generate_mode=="repeat":
            if image_id == self.append_id[-1]:
                #print("Using mode Image to generate image")
                init_image = Image.open(os.path.join(self.image_root, image_name))
                with torch.autocast('cuda'):
                    generate_image = self.model(
                        prompt=[caption],
                        mode="image",
                        height=512,
                        width=512,
                        num_inference_steps=50,
                        guidance_scale=self.guidance_scale,
                        init_image=init_image,
                        generator=self.generator,
                        strength=0.8,
                        return_intermediates=False,
                    )['sample']

            else:  # Case not repeat image
                #print("Using mode Prompt to generate image")
                        with torch.autocast('cuda'):
                            generate_image = self.model(
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
        else: 
            ## inCase the image name repeat 
            if image_id == self.append_id[-1]:
                # The first image repeat is creat
                image_name_= self.repeat_name[-1][:-5] + "1" +".jpg"
                path=os.path.join(self.image_root, image_name_)

                #checking image is exist or not
                if os.path.isfile(path) is True or os.path.exists(path) is True:
                    print("Next repeat image is append")
                    image_name= self.repeat_name[-1][:-4] + "1" + ".jpg"
                    image_id=self.append_id_repeat[-1] + "1"
                    path=os.path.join(self.image_root, image_name)
                    self.append_id_repeat.append(image_id)
                    self.repeat_name.append(image_name)

                else:
                    print("first repeat image is created")
                    image_id= image_id +"1"
                    image_name= image_name[:-4] + "1.jpg"
                    path=os.path.join(self.image_root, image_name)
                    self.append_id_repeat.append(image_id)
                    self.repeat_name.append(image_name)
        
            ## Append the new image name. 
            else: 
                self.append_id.append(image_id)
                self.append_id_repeat.append(image_id)
                self.repeat_name.append(image_name)

            with torch.autocast('cuda'):
                            generate_image = self.model(
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
        generate_image[0].save(path)
        self.append_id.append(image_id)
        print(f"image name {image_id} Generated")
        return image_name

    def save_json(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.new_json, outfile)

generate_data= COCO_synthetic_Dataset(image_root='/data/coco_synthetic/', ann_root='/data/coco_synthetic/', generate_mode="no_repeat")
## CoCo Caption dataset Caption Length 566.747=
print(generate_data.__len__())
for i in range(10000, 300000):
    generate_data.__getitem__(i)
generate_data.save_json("/data1/coco_synthetic_Dalle_SD/coco_synthetic_150k_200k.json") 
print("------------------------ Done ------------------------")

class COCO_synthetic_Dalle_SD(Dataset): 

    def __init__(self, image_root, ann_root, max_words=200, prompt='A photo of highly detailed of ', 
                    dalle_topk=128, temperature=2., supercondition_factor=16.,
                    guidance_scale=7.5,  num_inference_steps=50, seed=random.randint(50000, 1000000)):
        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'
        Path(image_root+ "val2014/").mkdir(parents=True, exist_ok=True)
        Path(image_root+ "train2014/").mkdir(parents=True, exist_ok=True)
        Path(image_root+ "dalle_image/").mkdir(parents=True, exist_ok=True)

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
            is_mega=False,  # False -> Using mini model,
            device='cuda',
            is_reusable=True,)

        ## Parameter for StableDiffusion Model 
        self.guidance_scale= guidance_scale
        self.num_inference_steps= num_inference_steps
        self.generator = torch.Generator(device="cuda").manual_seed(seed) #random.randint(0, 100000) #random.randint(0,10000) # change the seed to get different results
        self.append_id=["test"]
        self.append_id_repeat=["test"]
        self.repeat_name=['test']

        self.SD_model= StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,).to("cuda")
        self.new_json=[]
    
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        ann= self.annotation[idx]
        
        caption= self.prompt + pre_caption(ann['caption'], self.max_words)
        image_id= ann['image_id']
        image_name= ann['image'] ## Saved image's name
        
        ## Caption, value, name, value, image_name, value
        path=os.path.join(self.image_root, image_name)
   
        ## inCase the image name repeat 
        if image_id == self.append_id[-1]:
            # The first image repeat is creat
            image_name_= self.repeat_name[-1][:-5] + "1" +".jpg"
            path=os.path.join(self.image_root, image_name_)

            #checking image is exist or not
            if os.path.isfile(path) is True or os.path.exists(path) is True:
                print("Next repeat image is append")
                image_name= self.repeat_name[-1][:-4] + "1" + ".jpg"
                image_id=self.append_id_repeat[-1] + "1"
                path=os.path.join(self.image_root, image_name)
                self.append_id_repeat.append(image_id)
                self.repeat_name.append(image_name)

            else:
                print("first repeat image is created")
                image_id= image_id +"1"
                image_name= image_name[:-4] + "1.jpg"
                path=os.path.join(self.image_root, image_name)
                self.append_id_repeat.append(image_id)
                self.repeat_name.append(image_name)
    
        ## Append the new image name. 
        else: 
            self.append_id.append(image_id)
            self.append_id_repeat.append(image_id)
            self.repeat_name.append(image_name)
        
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
                print(init_image.size)
                init_image = init_image.resize((512, 512))
                generate_image= self.SD_model(
                                prompt=[caption],
                                mode="image",
                                height=512,
                               width=512,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                init_image=init_image,
                                generator=None, #self.generator,
                                strength=0.8,
                                return_intermediates=False,
                                )['sample']

       
        generate_image[0].save(path)
        #init_image.save(os.path.join("/data1/coco_synthetic/dalle_image/"+image_name))
        new_dict={"caption":ann['caption'],"image": image_name, "image_id": image_id}  
        self.new_json.append(new_dict)
        print(f"image name: {caption} Generated")
        return image_name 
        
    def save_json(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.new_json, outfile)

# generate_data= COCO_synthetic_Dalle_SD(image_root='/data1/coco_synthetic_Dalle_SD/', ann_root='/data1/coco_synthetic_Dalle_SD/')
# for i in range(150000, 200000):
#     generate_data.__getitem__(i)
# print("------------------------ Done ------------------------")
# generate_data.save_json("/data1/coco_synthetic_Dalle_SD/coco_synthetic_150k_200k.json")