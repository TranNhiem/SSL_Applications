from PIL import Image
import numpy as np 
import gradio as gr
import torch 
#from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import LMSDiscreteScheduler, DDIMScheduler, EulerDiscreteScheduler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SD_model="/data1/pretrained_weight/StableDiffusion"
img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    #use_auth_token=YOUR_TOKEN, 
    revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=SD_model,
         )
img_pipe.to(device)

def read_content(file_path) :
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

def resize(value,img):
  #baseheight = value
  img = Image.open(img)
  #hpercent = (baseheight/float(img.size[1]))
  #wsize = int((float(img.size[0])*float(hpercent)))
  #img = img.resize((wsize,baseheight), Image.Resampling.LANCZOS)
  img = img.resize((value,value), Image.Resampling.LANCZOS)
  return img

def infer(source_img, prompt,negative_promt, seed): 
    source_image= resize(512, source_img)
    source_image.save('source.png')
    generator = torch.Generator(device="cuda").manual_seed(int(seed)) # change the seed to get different results
    Euler_1 = EulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler", prediction_type="v_prediction")
    images_list = img_pipe([prompt] * 2, init_image=source_image, strength=0.75, scheduler=Euler_1,generator=generator).images
    return images_list

def run_demo(): 
    block = gr.Blocks(css=".container { max-width: 1300px; margin: auto; }")
    with block as demo:
        gr.HTML(read_content("header.html"))

        with gr.Row().style(mobile_collapse=False, equal_height=True):
            text = gr.Textbox(label="Your text prompt", placeholder="Typing: (what you want to edit in your image)..", show_label=True, max_lines=1).style( border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,)
            negative_promt= gr.Textbox(label="Context Information Removing", placeholder="Typing: (what you DON'T want in your image)..", show_label=True, max_lines=1).style( border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,)
        with gr.Row().style(mobile_collapse=False, equal_height=True):
            source_img = gr.Image(source="canvas", type="filepath", tool='color-sketch', label="new gradio color sketch")
        with gr.Row().style(mobile_collapse=False, equal_height=True):
            seed = gr.Number(value=12032, label="Different Image", show_label=True)
        with gr.Row().style(mobile_collapse=False, equal_height=True):
           btn= gr.Button(label="Generate", type="primary", id="generate")
        with gr.Row().style(mobile_collapse=False, equal_height=True):
            gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")
    
        btn.click(fn=infer, inputs=[source_img, text,negative_promt,seed ], outputs=[gallery ])
    
    demo.launch( share=False, enable_queue=True,  debug=True)  #server_name="172.17.0.1", # server_port=2222, share=True, enable_queue=True,  debug=True

if __name__ == '__main__':

    run_demo()