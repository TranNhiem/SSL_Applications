
 
import os
import sys 
import random
import torch
import PIL
import re
from PIL import Image
import numpy as np 
import gradio as gr
from torch import autocast
## Library for Stable diffusion inpainting model 
#from diffusion_pipeline.sd_inpainting_pipeline import  StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
import random
from glob import glob
from diffusers import LMSDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from torchvision import transforms
## API for Language Translation Model 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, GPT2Tokenizer, GPT2LMHeadModel,  set_seed
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

###--------------------------------
### Section Magic Prompt Generation
###--------------------------------
prompt_gen_model="/data1/pretrained_weight/prompt_gen"
# only cache the latest model
def get_model_and_tokenizer(model_id):
    model = GPT2LMHeadModel.from_pretrained(model_id, cache_dir=prompt_gen_model, device_map="auto") ## [load_in_8bit=True, torch_dtype=torch.float16]
    tokenizer = GPT2Tokenizer.from_pretrained(model_id,cache_dir=prompt_gen_model)
    return model, tokenizer


## Generate the Prompt with initial Prompt
model_id= "Gustavosta/MagicPrompt-Stable-Diffusion"
model, tokenizer =get_model_and_tokenizer(model_id)
prompt_="The image of beautiful mountain"
input_ids = tokenizer(prompt_, return_tensors='pt').input_ids

# generate the result with contrastive search
contrastive_sampling = model.generate(input_ids, penalty_alpha=0.6, top_k=4, max_length=random.randint(60, 90), num_return_sequences=1)
nucleus_sampling = model.generate(input_ids, do_sample=True, top_p=0.95, top_k=0, max_length=random.randint(60, 90), num_return_sequences=1)
beam_sampling=model.generate(input_ids, do_sample=True, num_beams=4, max_length=random.randint(60, 90), num_return_sequences=1)
greedy_sampling=model.generate(input_ids, do_sample=True,  max_length=random.randint(60, 90), num_return_sequences=1)

contrastive_text = tokenizer.decode(contrastive_sampling[0], skip_special_tokens=True)
nucleus_text = tokenizer.decode(nucleus_sampling[0], skip_special_tokens=True)
beam_text = tokenizer.decode(beam_sampling[0], skip_special_tokens=True)
greedy_text = tokenizer.decode(greedy_sampling[0], skip_special_tokens=True)
  
## Method 2 much cleaner with Transformer pipeline
# gpt2_pipe = pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')

# def generate(starting_text):
#     seed = random.randint(100, 1000000)
#     set_seed(seed)
#     output_1 = gpt2_pipe(starting_text, max_length=(len(starting_text) + random.randint(60, 90)),  num_return_sequences=2)# penalty_alpha=0.6, top_k=4,
#     #output_2 = gpt2_pipe(starting_text, max_length=(len(starting_text) + random.randint(60, 90)),top_k=4, num_return_sequences=2)
    
#     response_list = []
#     for x in output_1:
#         resp = x['generated_text'].strip()
#         if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
#             response_list.append(resp+'\n')
#     # for x in output_2:
#     #     resp = x['generated_text'].strip()
#     #     if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
#     #         response_list.append(resp+'\n')

#     response_end = "\n".join(response_list)
#     response_end = re.sub('[^ ]+\.[^ ]+','', response_end)
#     response_end = response_end.replace("<", "").replace(">", "")

#     if response_end != "":
#         return response_end
        
# text_output=generate(prompt_)

breakpoint()