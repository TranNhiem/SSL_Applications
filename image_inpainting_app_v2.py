
# @TranNhiem 2022 11/05
'''
Features Design include:
1.. Creating MASK using brush to draw mask or creating mask using ClipSeg model () 
2.. Text to Image Inpainting (Mask area or Background area)
3.. Prompting Using Language model for create prompt or User prompt input. 
4.. Creating Plugin exmaple style of Prompt and Upsampling for Inpainiting.

## Installations requirements 
!pip install -qq -U diffusers==0.6.0 transformers ftfy gradio
!pip install git+https://github.com/huggingface/diffusers.git

## Reference for Image Generation checkpoint model
Update Reference for Image Inpainting with Diffusion Models
https://huggingface.co/runwayml/stable-diffusion-inpainting 
https://huggingface.co/spaces/runwayml/stable-diffusion-inpainting/tree/main 


Note: You must comment these line of code from line 43 --> 49 of diffusers versio 0.6.0
    # mask = Image.fromarray(mask)
    # mask = np.array(mask.convert("L"))
    # mask = mask.astype(np.float32) / 255.0
    # mask = mask[None, None]
    # mask[mask < 0.5] = 0
    # mask[mask >= 0.5] = 1
    # mask = torch.from_numpy(mask)
   
"/anaconda/envs/solo_learn/lib/python3.10/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py"
'''
 
import os
import sys 
import random
import torch
import PIL
from PIL import Image
import numpy as np 
import gradio as gr
from torch import autocast
# import torchcsprng as csprng
#from diffusion_pipeline.sd_inpainting_pipeline import  StableDiffusionInpaintPipeline
from diffusers import StableDiffusionInpaintPipeline
from glob import glob
from diffusers import LMSDiscreteScheduler
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
lms = LMSDiscreteScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

pipeimg = StableDiffusionInpaintPipeline.from_pretrained(
    #"CompVis/stable-diffusion-v1-4",
    #"runwayml/stable-diffusion-v1-5", 
    "runwayml/stable-diffusion-inpainting",
     revision="fp16",
    torch_dtype=torch.float16,
    #scheduler=lms,
    #use_auth_token=True,
).to("cuda")

def dummy(images, **kwargs): return images, False
pipeimg.safety_checker = dummy


def mask_processes(mask, h,w):
    #mask=Image.fromarray(mask)
    # if w > 512:
    #     h = int(h * (512/w))
    #     w = 512
    # if h > 512:
    #     w = int(w*(512/h))
    #     h = 512
    # w, h = map(lambda x: x - x % 64, (w, h))
    # w //= 8
    # h //= 8
    # mask = mask.convert("L")
    # mask = mask.resize((w, h), resample=PIL.Image.LANCZOS)
    # # print(f"Mask size:, {mask.size}")
    # mask = np.array(mask).astype(np.float32) / 255.0
    # mask = np.tile(mask, (4, 1, 1))
    # mask = mask[None].transpose(0, 1, 2, 3)
    # mask[np.where(mask != 0.0)] = 1.0  # using bool to find the uer drawed
    # mask = torch.from_numpy(mask)

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    return mask


generator = torch.Generator(device="cuda").manual_seed(random.randint(0,10000)) # change the seed to get different results

def infer(prompt_, img, samples_num, steps_num, scale,option ): #option
    ## Checking the Image Size then Resize image
    #breakpoint()
    w, h=512,512
    # w, h = img["image"].size
    # if w > 512:
    #     h = int(h * (512/w))
    #     w = 512
    # if h > 512:
    #     w = int(w*(512/h))
    #     h = 512
    # w, h = map(lambda x: x - x % 64, (w, h))
    
    # w //= 8
    # h //= 8gr
    print("Your image Height and Width: ", h, w)
    mask = (img['mask'])
    mask = mask.resize((w, h),)# resample=PIL.Image.LANCZOS)
    mask=mask.convert("RGB")

    if option == "Background Area":
        #mask = (img['mask']).convert("RGB")
        mask = 1 - mask_processes(mask,  h, w)
        
        ## Squezing the first dimension of the tensor 
        # mask= torch.squeeze(mask, 0)
        # mask=transforms.ToPILImage()(mask)
    else:
        #mask = img['mask'].convert("RGB")
        mask = mask_processes(mask, h, w)
        # mask= torch.squeeze(mask, 0)
        # mask=transforms.ToPILImage()(mask)

    #breakpoint()
    # image = Image.fromarray(img['image'])
    # image=(image.resize((w,h), resample=PIL.Image.LANCZOS)).convert("RGB")

    image=(img['image']).resize((w,h), ).convert("RGB")#resample=PIL.Image.LANCZOS).convert("RGB")
    #image = image_preprocess(img['image'])
    #print("this is image shape", image.shape)

    print(prompt_)
    # Generate image for the masking area with prompt
    with autocast("cuda"):#"cuda"
    # with torch.cuda.amp.autocast(dtype=torch.float16):
        images = pipeimg([prompt_]*samples_num, image, mask,height=h, width=w,
                                num_inference_steps=steps_num, guidance_scale=scale, generator=generator).images# generator=generator
        return images[0]#[images]

def run_demo(): 

    block = gr.Blocks(css=".container { max-width: 1300px; margin: auto; }")
    with block as demo:
        gr.Markdown(
            "<h1><center> Image Inpainting 🎨 - 🖼️ </center></h1>")
        gr.HTML(
                    """
                        <div class="footer">
                            <p> INSTRUCTION to USE this App <a href=" https://youtu.be/q5kAOi-edoY" style="text-decoration: underline;" target="_blank"> Video Link </a> 
                            </p>
                        </div> 
                    """
                )
        with gr.Group():
            with gr.Box():
                with gr.Row().style(mobile_collapse=False, equal_height=True):
                    with gr.Column(scale=4, min_width=900, min_height=600):
                        text = gr.Textbox(label="Inpainting with Your text prompt", placeholder="Enter In-painting expected object here...", show_label=True, max_lines=1).style(
                            border=(True, False, True, True),
                            rounded=(True, False, False, True),
                            container=False,)
                    with gr.Column(scale=4, min_width=150, min_height=600):
                        btn = gr.Button("Run").style(
                            margin=False, rounded=(True, True, True, True),)

           

            # with gr.Row().style(mobile_collapse=False, equal_height=True):
            #     # option = gr.Radio(label=" Selecting Inpainting Area", default="Mask Area", choices=[
            #     #     "Mask Area", "Background Area"], show_label=True)
            #     option = gr.Dropdown( ["Replacing Mask Area", "Replacing Background Area"],label=" Choosing Replacing Part", show_label=True)
            
            with gr.Row().style(mobile_collapse=False, equal_height=True):

                with gr.Column(scale=1, min_width=80, min_height=400):
                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        image = gr.Image(source="upload", tool="sketch",label="Input image", type="pil").style(height=300)
                        #gallery = gr.Gallery(label="Generated images",show_label=True).style(grid=[2], height="auto")
                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                          # with gr.Row().style(mobile_collapse=False, equal_height=True):
       
                        option = gr.Dropdown( ["Mask Area", "Background Area"],label="Replacing Area", show_label=True)
            
                        samples_num = gr.Slider(label="Number of Generated Image",
                                                minimum=1, maximum=10, value=1, step=1,)  # show_label=False
                        steps_num = gr.Slider(
                            label="Generatio of steps", minimum=2, maximum=499, value=80, step=1,)  # show_label=False
                        scale = gr.Slider(label="Guidance scale", minimum=0.0,
                                        maximum=30, value=7.5, step=0.1,)  # show_label=False

                with gr.Column(scale=1, min_width=80, min_height=300):     
                    # text.submit(fn=infer, inputs=[
                    #             text, image, samples_num, steps_num, scale], outputs=[gr.Image()]) #[gr.Image()]
                    btn.click(fn=infer, inputs=[text, image, samples_num,
                            steps_num, scale, option], outputs=[gr.Image()])#[gr.Image()]

            # with gr.Row("20%"):
            #      gr.Markdown("#### (Illustration of Using This App):")
            #      gr.Image(f"{example_dir}/Rotating_earth_(large).gif", shape=(224, 224))
            # Using example images provide
            # with gr.Row("20%"):
            #     gr.Markdown("#### (Image Example):")
            #     #ims_uri = [f"{example_dir}/{x}" for x in os.listdir(example_dir)]##
            #     ims_uri=f"{example_dir}/343785.jpg"
            #     # ims_uri = [ex for ex in ims_uri]

            #     image = gr.Image(tool="sketch",label="Input image", type="numpy", value=str(ims_uri))#value=f"{example_dir}/343785.jpg") #source='upload',  # value=os.path.join(os.path.dirname(__file__),".jpg"
            #     gr.Examples(fn=infer, examples=[os.path.join(os.path.dirname(__file__)),image], inputs=[text, image, samples_num,steps_num, scale, option], outputs=gallery, cache_examples=True)

    demo.launch(server_name="0.0.0.0",  server_port=1234,
                share=True, enable_queue=True,  debug=True)  # debug=True)

if __name__ == '__main__':

    gr.close_all()
    run_demo()