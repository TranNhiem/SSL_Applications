import gradio as gr
import torch 
from PIL import Image
from utils import inversestablediffusion
from cross_attention import  AttentionStore, show_cross_attention, run_and_display, AttentionReplace, LocalBlend , AttentionRefine, get_equalizer, AttentionReweight
from diffusers import StableDiffusionPipeline, DDIMScheduler
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from utils import text2image_ldm_stable
from config import read_cfg
flag = read_cfg()
import torchvision.transforms as T

FLAGS = flag.FLAGS
#Init diffusion model
SD_Model = FLAGS.sd_model
model_path= FLAGS.store_path
#scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

ldm_stable = StableDiffusionPipeline.from_pretrained(SD_Model, cache_dir=model_path,).to(device)#scheduler= scheduler,
tokenizer = ldm_stable.tokenizer
MAX_NUM_WORDS = tokenizer.model_max_length
LOW_RESOURCE= FLAGS.low_resource
NUM_DIFFUSION_STEPS= FLAGS.num_diffusion_steps
GUIDANCE_SCALE= FLAGS.guidance_scale
print("Loaded all models to GPU")

generator= torch.cuda.manual_seed(798122)
input_image = Image.open("./portrait.jpg")
## Prompt form CLIP interrogator 
#prompt="a woman with long red hair and blue eyes, a computer rendering, photorealism, gorgeous female samara weaving, ana de armas as joan of arc, clean perfect symmetrical face, short blonde afro, redshift renderer, average human face, mid 2 0's female, an ai generated image"
prompt= "a photo of a woman with blonde hair"

init_latent= inversestablediffusion(ldm_stable,init_image=input_image, prompt=prompt, generator=generator, refine_iterations= 5, guidance_scale=4.0)
prompts=["a photo of a woman with blonde hair", "a photo of a woman with black hair"]
controller= AttentionReplace(tokenizer, prompts, NUM_DIFFUSION_STEPS, cross_replace_steps={"default": 1.,"lion": .4}, self_replace_steps=0.4)
image, latent= text2image_ldm_stable(ldm_stable,controller, prompts, latent= init_latent, num_inference_steps=NUM_DIFFUSION_STEPS, generator=generator, low_resource=LOW_RESOURCE)

image1=image[0]
image2=image[1]
transform = T.ToPILImage()
img = transform(image1)
img.save("new_img1.png")
img2 = transform(image2)
img2.save("new_img2.png")
breakpoint()
print(image.shape)

## LocalBlend (new args, tokenizer,MAX_NUM_WORDS )

##AttentionControl (LOW_RESOURCE=True)
## AttentionControlEdit(tokenizer)