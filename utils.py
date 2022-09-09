import numpy as np 
from PIL import Image
from io import BytesIO
import PIL
import torch 

# from dataclasses import dataclass
from dataclasses import field
from typing import Any, List,  Optional, Union
from pydantic.dataclasses import dataclass


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


@dataclass
class GeneratorConfig:
    """Configuration for a generation"""
    prompt: Union[str, List[str]]
    num_images: int = 1
    mode: str = "prompt"   # prompt, image, mask
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    eta: Optional[float] = 0.0
    # generator: Optional[Any] = None
    output_type: Optional[str] = "pil"
    strength: float = 0.8
    init_image: Any = None
    return_intermediates: bool = False