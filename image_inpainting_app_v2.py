import os
import sys 
import random
import torch
from torch import autocast
import gradio as gr
# import torchcsprng as csprng
from diffusion_pipeline.sd_inpainting_pipeline import  StableDiffusionInpaintPipeline
from glob import glob
from utils import mask_processes, image_preprocess
from diffusers import LMSDiscreteScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#lms = LMSDiscreteScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

pipeimg = StableDiffusionInpaintPipeline.from_pretrained(
    #"CompVis/stable-diffusion-v1-4",
    #"runwayml/stable-diffusion-v1-5", 
    "runwayml/stable-diffusion-inpainting",
     revision="fp16",
    torch_dtype=torch.float16,
    #use_auth_token=True,
).to("cuda")

def dummy(images, **kwargs): return images, False
pipeimg.safety_checker = dummy

generator = torch.Generator(device="cuda").manual_seed(random.randint(0,10000)) # change the seed to get different results
def infer(prompt_, img, samples_num, steps_num, scale, option):

    if option == "Background Area":
        #mask = (img['mask']).convert("RGB")
        mask = mask_processes(img['mask'])
    else:
        #mask = 1- (img['mask']).convert("RGB")
        mask = 1 - mask_processes(img['mask'])
        print("This is mask shape", mask.shape)

    #
    #image=(img['image']).convert("RGB")
    image = image_preprocess(img['image'])
    print("this is image shape", image.shape)

    print(prompt_)
    # Generate image for the masking area with prompt
    with autocast("cuda"):#"cuda"
    # with torch.cuda.amp.autocast(dtype=torch.float16):
        images = pipeimg([prompt_]*samples_num, image, mask,
                                num_inference_steps=steps_num, guidance_scale=scale, generator=generator).images# generator=generator
        return images[0]

# def run_demo(): 

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

        with gr.Row().style(mobile_collapse=False, equal_height=True):
            samples_num = gr.Slider(label="Number of Generated Image",
                                    minimum=1, maximum=10, value=1, step=1,)  # show_label=False
            steps_num = gr.Slider(
                label="Generatio of steps", minimum=2, maximum=499, value=80, step=1,)  # show_label=False
            scale = gr.Slider(label="Guidance scale", minimum=0.0,
                            maximum=30, value=7.5, step=0.1,)  # show_label=False

        with gr.Row().style(mobile_collapse=False, equal_height=True):
            # option = gr.Radio(label=" Selecting Inpainting Area", default="Mask Area", choices=[
            #     "Mask Area", "Background Area"], show_label=True)
            option = gr.Dropdown( ["Replacing Mask Area", "Replacing Background Area"],label=" Choosing Replacing Part", show_label=True)
        
        image = gr.Image(source="upload", tool="sketch",label="Input image", type="numpy").style(height=400)
        gallery = gr.Gallery(label="Generated images",show_label=True).style(grid=[2], height="auto")
        text.submit(fn=infer, inputs=[
                    text, image, samples_num, steps_num, scale, option], outputs=gallery)

        btn.click(fn=infer, inputs=[text, image, samples_num,
                steps_num, scale, option], outputs=gallery)

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

demo.launch(server_name="0.0.0.0",  server_port=12345,
            share=True, enable_queue=True,  debug=True)  # debug=True)

# if __name__ == '__main__':

#     gr.close_all()
#     run_demo()