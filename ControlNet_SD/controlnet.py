'''
TranNhiem 2023/03/02 

    Support Features: 
        + 

    Expected Product for Service: 
        + Kids Education App 
        + Increasing Art Creation 
        + Home Design and Architecture Company for Home Design adding New Product 


Reference: 
    https://github.com/lllyasviel/ControlNet/blob/main/docs/annotator.md 
    https://huggingface.co/spaces/hysts/ControlNet 

'''

import sys
sys.path.append('/home/rick/SSL_Application/SSL_Applications/ControlNet_SD/ControlNet')

import pathlib
import random
import shlex
import subprocess
import cv2 
import einops 
import numpy as np
import torch 
from pytorch_lightning import seed_everything
from ControlNet import config 

from ControlNet.annotator.canny import apply_canny
from ControlNet.annotator.hed import apply_hed, nms
from ControlNet.annotator.midas import apply_midas
from ControlNet.annotator.mlsd import apply_mlsd
from ControlNet.annotator.openpose import apply_openpose
from ControlNet.annotator.uniformer import apply_uniformer
from ControlNet.annotator.util import HWC3, resize_image
from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.ldm.models.diffusion.ddim import DDIMSampler

from ControlNet.share import * 

## Base Model 
ORIGINAL_WEIGHT_ROOT = 'https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/'
ORIGINAL_MODEL_NAMES = {
    'canny': 'control_sd15_canny.pth',
    'hough': 'control_sd15_mlsd.pth',
    'hed': 'control_sd15_hed.pth',
    'scribble': 'control_sd15_scribble.pth',
    'pose': 'control_sd15_openpose.pth',
    'seg': 'control_sd15_seg.pth',
    'depth': 'control_sd15_depth.pth',
    'normal': 'control_sd15_normal.pth',
}
## LightWeight Model --> Optimization for Mobile Device
LIGHTWEIGHT_MODEL_NAMES = {
    'canny': 'control_canny-fp16.safetensors',
    'hough': 'control_mlsd-fp16.safetensors',
    'hed': 'control_hed-fp16.safetensors',
    'scribble': 'control_scribble-fp16.safetensors',
    'pose': 'control_openpose-fp16.safetensors',
    'seg': 'control_seg-fp16.safetensors',
    'depth': 'control_depth-fp16.safetensors',
    'normal': 'control_normal-fp16.safetensors',
}
LIGHTWEIGHT_WEIGHT_ROOT = 'https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/'

## Class Object Using Model 
class ControlNet:
    def __init__(self,
                 
                 model_config_path: str = 'ControlNet/models/cldm_v15.yaml',
                 model_dir: str = 'models',# Output Path to store weights 
                 use_lightweight: bool = True):
                

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model(model_config_path).to(self.device)
        self.ddim_sampler = DDIMSampler(self.model)
        self.task_name = ''

        self.model_dir = pathlib.Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)

        self.use_lightweight = use_lightweight
        
        ## Load Model 
        if self.use_lightweight:
            self.model_names = LIGHTWEIGHT_MODEL_NAMES
            self.weight_root = LIGHTWEIGHT_WEIGHT_ROOT
            base_model_url = 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors'
            self.load_base_model(base_model_url)
        else:
            self.model_names = ORIGINAL_MODEL_NAMES
            self.weight_root = ORIGINAL_WEIGHT_ROOT
        
        self.download_models() 
    
    def download_base_model(self, model_url: str) -> pathlib.Path:
        model_name = model_url.split('/')[-1]
        model_path = self.model_dir / model_name
        if not model_path.exists():
            print(f'Downloading {model_name} ...')
            subprocess.run(shlex.split(f'wget {model_url} -P {self.model_dir}'))
        return model_path
    
    def load_base_model(self, model_url: str):
        model_path = self.download_base_model(model_url)
        self.model.load_state_dict(load_state_dict(model_path, location=self.device.type), strict=False)
    
    def load_weight(self, task_name: str) -> None: 
        if task_name ==self.task_name: 
            return 
        weight_path= self.get_weight_path(task_name)
        if not self.use_lightweight:
            self.model.load_state_dict(load_state_dict(weight_path, location=self.device.type), strict=False)
            