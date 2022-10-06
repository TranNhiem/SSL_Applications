import numpy as np 
from PIL import Image
from io import BytesIO
import PIL
import torch 
from torch import nn 
from torch.nn import functional as F 
# from dataclasses import dataclass
from dataclasses import field
from typing import Any, List,  Optional, Union
# from pydantic.dataclasses import dataclass



def preprocess_pil(image):
    w, h = image.size
    if w > 512:
        h = int(h * (512/w))
        w = 512
    if h > 512:
        w = int(w*(512/h))
        h = 512
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

## Processing RGB image Handle image size for Small GPU V Ram
def image_preprocess(image, full_resolution=False):
    image = Image.fromarray(image)
    w, h = image.size
    
    ## Consider to comment these line keeping Full image resolution
    if full_resolution: 
        w, h= w,h
    else: 
        if w > 512:
            h = int(h * (512/w))
            w = 512
        if h > 512:
            w = int(w*(512/h))
            h = 512
    # resize to integer multiple of 64, 32 can sometimes result in tensor mismatch errors
    w, h = map(lambda x: x - x % 64, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    print(f"this is image.size: {image.size}")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def mask_processes(mask):
    mask = Image.fromarray(mask)
    mask = mask.convert("L")
    w, h = mask.size
    if w > 512:
        h = int(h * (512/w))
        w = 512
    if h > 512:
        w = int(w*(512/h))
        h = 512
    w, h = map(lambda x: x - x % 64, (w, h))
    w //= 8
    h //= 8

    mask = mask.resize((w, h), resample=PIL.Image.LANCZOS)
    print(f"Mask size:, {mask.size}")
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)
    mask[np.where(mask != 0.0)] = 1.0  # using bool to find the uer drawed
    mask = torch.from_numpy(mask)
    return mask

# @dataclass
# class GeneratorConfig:
#     """Configuration for a generation"""
#     prompt: Union[str, List[str]]
#     num_images: int = 1
#     mode: str = "prompt"   # prompt, image, mask
#     height: Optional[int] = 512
#     width: Optional[int] = 512
#     num_inference_steps: Optional[int] = 50
#     guidance_scale: Optional[float] = 7.5
#     eta: Optional[float] = 0.0
#     # generator: Optional[Any] = None
#     output_type: Optional[str] = "pil"
#     strength: float = 0.8
#     init_image: Any = None
#     return_intermediates: bool = False


#--------------------------------------------------------------------
# Section for Visual Grounding Language - Visual Helper function  
#--------------------------------------------------------------------

def coord2bin(coords, w_resize_ratio, h_resize_ratio,task): 
    coord_list= [float(coord) for coord in coords.strip().split()]
    bin_list = [] 
    bin_list += ["<bin_{}>".format(int(round(coord_list[0] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[1] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[2] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[3] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    return ' '.join(bin_list)

## Getting coordinate f
def bin2coord(bins, w_resize_ratio, h_resize_ratio, task): 
    bin_list= [int(bin[5:-1]) for bin in bins.strip().split()]
    coord_list= [] 
    coord_list += [bin_list[0] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[1] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    coord_list += [bin_list[2] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[3] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    return coord_list 

def get_symbols_to_strip_from_output(generator): 
    if hasattr(generator, "symbols_to_strip_from_output"): 
        return generator.symbols_to_strip_from_output
    else: 
        return {generator.bos, generator.eos}

def decode_fn(x, tgt_dict, bpe, generator, tokenizer= None): 
    x= tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    token_result= [] 
    bin_result= [] 
    img_result = [] 
    for token in x.strip().split(): 
        if token.startswith('<bin_'): 
            bin_result.append(token)
        elif token.startswith('<code_'): 
            img_result.append(token)
        else: 
            if bpe is not None: 
                token= bpe.decode('{}'.format(token))
            if tokenizer is not None: 
                token= tokenizer.decode(token)
            if token.startswith('') or len(token_result) == 0: 
                token_result.append(token.strip())
            else: 
                token_result[-1] += token 
    return ' '.join(token_result), ' '.join(bin_result), ' '.join(img_result)

def preprocess_mask_v2(mask):
    mask=mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w//8, h//8), resample=PIL.Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask,(4,1,1))
    mask = mask[None].transpose(0, 1, 2, 3)#what does this step do?
    mask = 1 - mask #repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask

def preprocess_image_v2(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


#--------------------------------------------------------------------
# Section Stable Diffusion Guided CLIP Helper function
#--------------------------------------------------------------------
#The class object to resize the image generated feature.
class resize_feature(nn.Module): 
    def __init__(self, cut_size, cut_power=1.0): 
        super().__init__()
        self.cut_size = cut_size
        self.cut_power = cut_power 
    
    def forward(self, pixel_values, num_cutouts): 
        sideY, sideX = pixel_values.shape[2:4]
        max_size= min(sideX, sideY)
        min_size= min(sideX, sideY, self.cut_size)
        cutouts= [] 
        
        for _ in range(num_cutouts): 
            size= int(torch.rand([])**self.cut_power * (max_size - min_size) + min_size)
            offsetx= torch.randint(0, sideX - size + 1, ())
            offsety= torch.randint(0, sideY - size + 1, ())
            cutout= pixel_values[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout,  self.cut_size))
        
        return torch.cat(cutouts)

def set_requires_grad(model, requires_grad): 
    for param in model.parameters(): 
        param.requires_grad= requires_grad 

def image_grid(imgs, rows, cols): 
    assert len(imgs)== rows*cols 
    w, h = imgs[0].size
    grid= Image.new('RGB', size=(cols*w, rows*h))
    grid_w, gird_h = grid.size 

    for i, img in enumerate(imgs): 
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid 


