import gradio as gr
import torch 
from cross_attention import AttentionStore, show_cross_attention, run_and_display, AttentionReplace, LocalBlend , AttentionRefine, get_equalizer, AttentionReweight
import utils
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from absl_mock import Mock_Flag 
from config import read_cfg


store_path="/data1/pretrained_weight/StableDiffusion/"
pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=True,
            cache_dir= store_path, 
        )
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
        





def inference(model, img, mask, steps=1000, lr=0.1, sigma=0.1, noise=0.1, 
              scheduler=None, diffuser=None, 
              attention_store=None, attention_replace=None, 
              attention_refine=None, attention_reweight=None, 
              local_blend=None, equalizer=None):
    if scheduler is None:
        scheduler = DDIMScheduler(steps=steps, lr=lr, sigma=sigma, noise=noise)
    if diffuser is None:
        diffuser = StableDiffusionPipeline(scheduler=scheduler)
    if attention_store is None:
        attention_store = AttentionStore()
    if attention_replace is None:
        attention_replace = AttentionReplace()
    if attention_refine is None:
        attention_refine = AttentionRefine()
    if attention_reweight is None:
        attention_reweight = AttentionReweight()
    if local_blend is None:
        local_blend = LocalBlend()
    if equalizer is None:
        equalizer = get_equalizer()

        
    return run_and_display(model, img, mask, diffuser, scheduler, 
                           attention_store, attention_replace, 
                           attention_refine, attention_reweight, 
                           local_blend, equalizer)

