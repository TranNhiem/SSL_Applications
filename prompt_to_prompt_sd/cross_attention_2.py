'''
TranNhiem 2023/01/01

Features Design include:
This build on Version 1 of Prompt to Prompt SD 

****************Use case application:****************
Pipeline: 1 input prompt -> 2 prompt Image ->  3 Edit Prompt (expecting some changes) -> 4 output Image
Pipeline: 2 Initial Image  -> 2 Captioning use as Initial Prompt -> 3 Editting the init prompt -> 4 output Image
Null-Text Version 
************Supporting Feature Design:****************
1.. Creating Image Inversion from Init prompt Image with Prompt editing  
2.. Reducing the Weight attention of the object in the image (Prompt editing)
3.. Target replacement with (Prompt editing) 
4.. Style Transfer with (Prompt editing) 

************References from Github Repo:****************
https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images

'''

from typing import Optional, Union, Tuple, List, Callable, Dict
import numpy as np
from torch.nn import functional as nnf
import abc
import torch 
import utils
import seq_aligner

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOW_RESOURCE= True 


class LocalBlend: 

    def get_mask(self, maps, alpha, use_pool): 
        k=1 
        maps= (maps* alpha).sum(-1).mean(1)

        if use_pool:
            maps= nnf.max_pool2d(maps, kernel_size=3, stride=1, padding=1)
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t,attention_store):
        self.counter +=1 
        if self.counter > self.start_blend: 
            maps= attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps= [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, self.MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])

        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]],tokenizer,MAX_NUM_WORDS,  substruct_words=None, start_blend=0.2, th=(.3, .3)): 
        
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)

        for i , (prompt, words_) in enumerate(zip(prompts, words)):
            
            if type(words_) == str:
                words_ = [words_]

            for word in words_:
                ind= utils.get_word_inds(prompt,word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        if substruct_words is not None: 
            substruct_layers= torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
            for i , (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) == str:
                    words_ = [words_]

                for word in words_:
                    ind= utils.get_word_inds(prompt,word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers= substruct_layers.to(device)

        else: 
            self.substruct_layers = None
        self.alpha_layers= alpha_layers.to(device)
        self.start_blend= int(start_blend* MAX_NUM_WORDS)
        self.counter=0 
        self.th= th
        self.MAX_NUM_WORDS= MAX_NUM_WORDS


class EmptyControl: 
    def step_callback(self, x_t): 
        return x_t
    
    def between_steps(self): 
        return 
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str): 
        return attn 

class AttentionControl(abc.ABC): 

    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0 
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet:str): 
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str): 
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE: 
                attn= self.forward(attn, is_cross, place_in_unet)
            
            else: 
                h= attn.shape[0]
                attn[h//2:] = self.forward(attn[h//2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
    
    def __init__(self,): 
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl): 
    def step_callback(self, x_t): 
        if self.cur_step < self.stop_inject: 
            b= x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t
    
    def __init__(self, stop_inject:float, num_step: int): 
        super(SpatialReplace, self).__init__()
        self.stop_inject= int((1- stop_inject) * num_step)

class AttentionStore(AttentionControl): 
    @staticmethod
    def get_empty_store(): 
        return {"down_cross": [],"mid_cross": [], "up_cross": [], 
                "down_self": [], "mid_self": [], "up_self": []}
    
    def forward(self, attn, is_cross: bool, place_in_unet: str): 
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self): 
        if len(self.attention_store) == 0: 
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

    def __init__(self,):
        super(AttentionStore, self).__init__()
        self.step_store= self.get_empty_store()
        self.attention_store = {}

class AttentionControlEdit(AttentionStore, abc.ABC): 
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base

        else: 
            return att_replace 

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace, place_in_unet):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet) 
        

        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]): 
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_replace = attn[0], attn[1:]

            if is_cross: 
                alpha_words= self.cross_replace_alpha[self.cur_step]
                attn_replace_new= self.replace_cross_attention(attn_base, attn_replace) * alpha_words + (1- alpha_words)* attn_replace
                attn[1:] = attn_replace_new

            else: 
                attn[1:] = self.replace_self_attention(attn_base, attn_replace, place_in_unet)
            attn= attn.reshape(self.batch_size *h, *attn.shape[2:])
        
        return attn 
    
    def __init__(self, prompts, num_steps: int, cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]], 
        self_replace_steps: Union[float, Tuple[float, float]],local_blend: Optional[LocalBlend], tokenizer,):


        super(AttentionControlEdit, self).__init__()
        self.batch_size= len(prompts)
        self.cross_replace_alpha= utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps =0, self_replace_steps 
        
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend= local_blend 


class AttentionReplace(AttentionControlEdit): 
    def replace_cross_attention(self, attn_base, att_replace): 
        return torch.einsum('hpw, bwn -> bhpn', attn_base, self.mapper)

    def __init__(self, prompts, tokenizer, num_steps: int, cross_replace_steps: float, 
                    self_replace_steps: float, local_blend: Optional[LocalBlend]=None): 
        super(AttentionReplace, self).__init__(prompts, tokenizer, num_steps, cross_replace_steps, self_replace_steps, local_blend):
        self.mapper= seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)

class AttentionRefine(AttentionControlEdit): 
    
    def replace_cross_attention(self, attn_base, att_replace): 

        attn_base_replace= attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace= attn_base_replace * self.alphas + att_replace * (1- self.alphas)

        return attn_replace 
    
    def __init__(self, prompts, tokenizer, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, tokenizer, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas= seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit): 

    def replace_cross_attention(self, attn_base, att_replace): 
        if self.prev_controller is not None: 
            attn_base= self.prev_controller(att_replace, is_cross= True, place_in_unet= None)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]

        return attn_replace

    def __init__(self, prompts, tokenizer, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer, 
                 local_blend: Optional[LocalBlend] = None, prev_controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, tokenizer, num_steps, cross_replace_steps, self_replace_steps, local_blend)
    
        self.equalizer= equalizer
        #self.equalizer = seq_aligner.get_equalizer(prompts, tokenizer).to(device)
        self.prev_controller= prev_controller

def get_equalizer(text: str, tokenizer, word_select: Union[int, Tuple[int, ...]], values: Union[float, Tuple[float, ...]]): 
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer= torch.ones(1, tokenizer.model_max_length)
    for word, val in zip(word_select, values): 
        inds= utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds]= val

    return equalizer

def aggregate_attention(attention_store: AttentionStore, res:int, from_where: List[str], is_cross: bool, select: int): 
    out= []
    attention_maps= attention_store.get_average_attention()
    num_pixels= res**2 
    for location in from_where: 
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]: 
            if item.shape[1] == num_pixels: 
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out =torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]

    return out.cpu() 

def make_controller(prompts: List[str],tokenizer, num_steps: int,  is_replace_controller: bool, 
                    cross_replace_steps: Dict[str, float], self_replace_steps: float
                    ,blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else: 
        lb = LocalBlend(prompts, blend_words)
    
    if is_replace_controller:
        controller = AttentionReplace(prompts,tokenizer, num_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, tokenizer, num_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], tokenizer,equilizer_params["words"], equilizer_params["values"])
        controller= AttentionReweight(prompts, tokenizer, num_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, prev_controller=controller)

    return controller


def show_cross_attention(tokenizer, attention_store: AttentionStore, res: int, from_where: List[str], select: int=0): 
    tokens= tokenizer.encode(prompts[select])


          
    



    



