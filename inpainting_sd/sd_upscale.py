'''
@TranNhiem 2022 
Using K diffusion with Text Condition to Upscale an image
'''

import k_diffusion as K
import torch
import torch.nn.functional as F
from torch import nn

class NoiseLevelAndTextConditionedUpscaler(nn.Module):
    def __init__(self, inner_model, sigma_data=1., embed_dim=256): 
        super().__init__()
        self.inner_model= inner_model 
        self.sigma_data= sigma_data
        self.low_res_noise_embed= K.layers.FourierFeatures(1, embed_dim, std=2)


    def forward(self, input, sigma, low_res, low_res_sigma, c, **kwargs): 
        cross_cond, cross_cond_padding, pooler= c
        c_in = 1 / (low_res_sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_noise = low_res_sigma.log1p()[:, None]
        c_in = K.utils.append_dims(c_in, low_res.ndim)
        low_res_noise_embed = self.low_res_noise_embed(c_noise)
        low_res_in = F.interpolate(
            low_res, scale_factor=2, mode='nearest') * c_in
        
        mapping_cond = torch.cat([low_res_noise_embed, pooler], dim=1)
        return self.inner_model(input, sigma, unet_cond=low_res_in, mapping_cond=mapping_cond, cross_cond=cross_cond, cross_cond_padding=cross_cond_padding, **kwargs)

def make_upscaler_model(config_path, model_path, )
