
# @TranNhiem 2022 11/21
'''
Features Design include:
This build on Version 1 of Prompt to Prompt SD 

****************Use case application:****************
Pipeline: 1 input prompt -> 2 prompt Image ->  3 Edit Prompt (expecting some changes) -> 4 output Image
Pipeline: 2 Initial Image  -> 2 Captioning use as Initial Prompt -> 3 Editting the init prompt -> 4 output Image

************Supporting Feature Design:****************
1.. Creating Image Inversion from Init prompt Image with Prompt editing  
2.. Reducing the Weight attention of the object in the image (Prompt editing)
3.. Target replacement with (Prompt editing) 
4.. Style Transfer with (Prompt editing) 

'''


import numpy as np
import random
from PIL import Image
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from difflib import SequenceMatcher
from IPython.display import display
import requests
from io import BytesIO
from typing import Optional, Union, Tuple, List, Callable, Dict
import utils
import seq_aligner

import torch.nn.functional as nnf
import abc
import torch
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


#### Update Using the Larges OpenCLIP Model
#Init CLIP tokenizer and model
model_path_clip = "openai/clip-vit-large-patch14"
#Init diffusion model
SD_Model = "CompVis/stable-diffusion-v1-4"
model_path= "/data1/pretrained_weight/StableDiffusion/"

clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip,  cache_dir= model_path)
clip_model = CLIPModel.from_pretrained(model_path_clip,  cache_dir= model_path,  torch_dtype=torch.float16)
clip = clip_model.text_model
MAX_NUM_WORDS = clip_tokenizer.model_max_length
LOW_RESOURCE = False ## True for Runing on 12GB GPU

unet = UNet2DConditionModel.from_pretrained(SD_Model,  cache_dir= model_path, subfolder="unet", revision="fp16", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(SD_Model, cache_dir= model_path,  subfolder="vae", revision="fp16", torch_dtype=torch.float16)

unet.to(device)
vae.to(device)
clip.to(device)
print("Loaded all models to GPU")


class LocalBlend: 
    '''
    AttentionControl object. 
    The forward pass is called in each attention layer of the diffusion model and it can modify the input [ATTENTION WEIGHT] attn.
    is_cross, place_in_unet in ("down", "mid", "up"), 
    AttentionControl.cur_step help us track the exact 
    1 attention layer  
    2 timestamp during the diffusion iference.
    '''
    def __call__(self, x_t, attention_store): 
        k=1 
        maps= attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps= [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps= torch.cat(maps, dim=1)
        maps= (maps * self.alpha_layers).sum(-1).mean(1)
        mask= nnf.max_pool2d(maps, (k*2+1, k*2 +1), (1, 1), padding=(k, k))
        mask= nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask= (mask[:1] + mask[1:]).float()
        x_t= x_t [:1] + mask * (x_t - x_t[:1])

        return x_t 

    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3): 
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = utils.get_word_inds(prompt, word, clip_tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold

class AttentionControl(abc.ABC):
    def step_callback(self, x_t): 
        return x_t 
    def between_steps(self): 
        return 
    
    @property 
    def num_uncond_att_layers(self): 
        return self.num_att_layers if LOW_RESOURCE else 0 

    @abc.abstractmethod 
    def forward(self, attn, is_cross: bool, place_in_Unet: str): 
        raise NotImplementedError 
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str): 
        if self.cur_att_layer >= self.num_uncond_att_layers: 

            if LOW_RESOURCE: 
                attn= self.forward(attn, is_cross, place_in_unet)
            else: 
                h= attn.shape[0]
                attn[h // 2: ] = self.forward(attn[h //2:], is_cross, place_in_unet )
        
        self.cur_att_layer += 1 
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers: 
            self.cur_att_layer = 0 
            self.cur_step +=1 
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class AttentionStore(AttentionControl): 
    @staticmethod 
    def get_empty_store(): 
        return {"down_cross": [], "down_self": [], "mid_cross": [], "mid_self": [], "up_cross": [], "up_self": []}
    
    def forward(self, attn, is_cross: bool, place_in_unet: str): 
        key = f"{place_in_unet}_{'' if is_cross else 'self'}"
        if attn.shape[1] <= 32 **2: 
            self.step_store[key].append(attn)
        return attn 

    def between_steps(self):
        if len(self.attention_store)== 0: 
            self.attention_store= self.step_store

        else: 
            for key in self.attention_store: 
                for i in range(len(self.attention_store[key])): 
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store= self.get_empty_store()

    def get_average_attention(self): 
        average_attention= {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention 

    def reset(self): 
        super(AttentionStore, self).reset() 
        self.step_store= self.get_empty_store()
        self.attention_store = {} 

class AttentionControlEdit(AttentionStore, abc.ABC): 

    def step_callback(self, x_t): 
        if self.local_blend is not None: 
            x_t= self.local_blend(x_t, self.attention_store)
            return x_t 
    def replace_self_attention(self, attn_base, att_replace): 
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int, cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]], 
                self_replace_steps: Union[float, Tuple[float, float]], 
                local_blend: Optional[LocalBlend]
            ): 
        super(AttentionControlEdit, self).__init__() 
        self.batch_size= len(prompts)
        self.cross_replace_alpha= utils.get_time_words_attention_apha(prompts, num_steps, cross_replace_steps, clip_tokenizer).to(device)

class AttentionReplace(AttentionControlEdit): 
    def replace_cross_attention(self, attn_base, att_replace): 
        return torch.einsum('hpw, bwn -> bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, local_blend: Optional[LocalBlend]= None ): 
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, clip_tokenizer).to(device)


class AttentionRefine(AttentionControlEdit): 
    
    def replace_cross_attention(self, attn_base, att_replace): 
        attn_base_replace= attn_base[:, :, self.mapper].permute(2, 0, 1, 3 )
        attn_replace=  attn_base_replace.unsqueeze(0) * self.alphas + att_replace * (1- self.alphas)
        return attn_replace 
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, local_blend: Optional[LocalBlend]= None ): 
        
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, clip_tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas= alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

class AttentionReweight(AttentionControlEdit): 

    def replace_cross_attention(self, attn_base, att_replace): 
        if self.prev_controller is not None: 
            attn_base= self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace= attn_base [None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer, 
                local_blend: Optional[LocalBlend]=None, controller: Optional[AttentionControlEdit]= None): 
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer= equalizer.to(device)
        self.prev_controller= controller 

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float]], Tuple[float, ...]): 
    if type(word_select) is int or type(word_select) is str: 
        word_select= (word_select,)
    equalizer= torch.ones(len(values), MAX_NUM_WORDS)
    values= torch.tensor(values, dtype=torch.float32)
    for word in word_select: 
        inds= ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values 
    return equalizer 


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str])