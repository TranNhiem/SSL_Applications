
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline, DiffusionPipeline, StableDiffusionDepth2ImgPipeline, DPMSolverMultistepScheduler
import gradio as gr
import torch
from PIL import Image
import random
import math 
#--------------------------------------
### Section for for loading Loading TEXT-IMAGE model
#--------------------------------------

# model_id = 'stabilityai/stable-diffusion-2'
model_id = 'stabilityai/stable-diffusion-2-1'
#scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
SD_model="/data1/pretrained_weight/StableDiffusion/"
DPM_Solver = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    trained_betas=None,
    predict_epsilon=True,
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,)

pipe = StableDiffusionPipeline.from_pretrained(
      model_id,
      revision="fp16" if torch.cuda.is_available() else "fp32",
      torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
      scheduler=DPM_Solver, 
      cache_dir=SD_model,
    ).to("cuda")
pipe.enable_attention_slicing()
# pipe.enable_xformers_memory_efficient_attention()


#--------------------------------------
### Section for for loading Loading UPSCALE-IMAGE model
#--------------------------------------
def get_upscale_pipe(scheduler, cache_dir):
    
    update_state("Loading upscale model...")

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
      revision="fp16" if torch.cuda.is_available() else "fp32",
      torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
      scheduler=scheduler,
      cache_dir=cache_dir,
    )
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    set_mem_optimizations(pipe)
    pipe.to("cuda")
    return pipe

#--------------------------------------

attn_slicing_enabled = True
mem_eff_attn_enabled = False
state = None
current_steps = 25

pipe_upscale = None
modes = {
    'txt2img': 'Text to Image',
    'inpaint': 'Inpainting',
    'upscale4x': 'Upscale 4x',
}
current_mode = modes['txt2img']

def error_str(error, title="Error"):
    return f"""#### {title}
            {error}"""  if error else ""

def update_state(new_state):
  global state
  state = new_state

def update_state_info(old_state):
  if state and state != old_state:
    return gr.update(value=state)

def set_mem_optimizations(pipe):
    if attn_slicing_enabled:
      pipe.enable_attention_slicing()
    else:
      pipe.disable_attention_slicing()


def switch_attention_slicing(attn_slicing):
    global attn_slicing_enabled
    attn_slicing_enabled = attn_slicing

def switch_mem_eff_attn(mem_eff_attn):
    global mem_eff_attn_enabled
    mem_eff_attn_enabled = mem_eff_attn

def pipe_callback(step: int, timestep: int, latents: torch.FloatTensor):
    update_state(f"{step}/{current_steps} steps")#\nTime left, sec: {timestep/100:.0f}")

def inference(inf_mode, prompt, n_images, guidance, steps, width=768, height=768, seed=0, img=None, strength=0.5, neg_prompt="", upscale=4):

    update_state(" ")
    global current_mode
    if inf_mode != current_mode:
        pipe.to("cuda" if inf_mode == modes['txt2img'] else "cpu")
        
        if pipe_upscale is not None:
            pipe_upscale.to("cuda" if inf_mode == modes['upscale4x'] else "cpu")
    
        current_mode = inf_mode
    
    if seed == 0:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator('cuda').manual_seed(seed)
    prompt = prompt

    try:
    
        if inf_mode == modes['txt2img']:
            return txt_to_img(prompt, n_images, neg_prompt, guidance, steps, width, height, generator, seed), gr.update(visible=False, value=None)

        elif inf_mode == modes['upscale4x']:
            if img is None:
                return None, gr.update(visible=True, value=error_str("Image is required for Upscale mode"))

            return upscale(prompt, n_images, neg_prompt, img, guidance, steps, generator,upscale), gr.update(visible=False, value=None)

    except Exception as e:
        return None, gr.update(visible=True, value=error_str(e))

def txt_to_img(prompt, n_images, neg_prompt, guidance, steps, width, height, generator, seed):


    result = pipe(
      prompt,
      num_images_per_prompt = n_images,
      negative_prompt = neg_prompt,
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator,
      callback=pipe_callback).images

    update_state(f"Done. Seed: {seed}")

    return result

def upscale(prompt, n_images, neg_prompt, img, guidance, steps, generator, upscale): 
    global pipe_upscale
    if pipe_upscale is None:
        pipe_upscale = get_upscale_pipe(DPM_Solver, SD_model)


    img = img['image']
    return upscale_tiling(prompt, neg_prompt, img, guidance, steps, generator, upscale)


    # result = pipe_upscale(
    #   prompt,
    #   num_images_per_prompt = n_images,
    #   negative_prompt = neg_prompt,
    #   num_inference_steps = int(steps),
    #   guidance_scale = guidance,
    #   image = img,
    #   generator = generator,
    #   callback=pipe_callback).images

    # update_state("Done.")

    # return result
def upscale_tiling(prompt, neg_prompt, img, guidance, steps, generator, upscale=4): 
    width, height = img.size
    # calculate the padding needed to make the image dimensions a multiple of 128
    padding_x = 128 - (width % 128) if width % 128 != 0 else 0
    padding_y = 128 - (height % 128) if height % 128 != 0 else 0
    
    # create a white image of the right size to be used as padding
    padding_img = Image.new('RGB', (padding_x, padding_y), color=(255, 255, 255, 0))

    # paste the padding image onto the original image to add the padding
    img.paste(padding_img, (width, height))
    ## Update the image dimensions to include the padding 
    width += padding_x
    height += padding_y

    if width > 128 or height > 128: 
        num_tiles_x = math.ceil(width / 128)
        num_tiles_y = math.ceil(height / 128)
        upscaled_img = Image.new('RGB', (img.size[0] * upscale, img.size[1] * upscale))
        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                update_state(f"Upscaling tile {x * num_tiles_y + y + 1}/{num_tiles_x * num_tiles_y}")
                tile = img.crop((x * 128, y * 128, (x + 1) * 128, (y + 1) * 128))   
                upscaled_tile = pipe_upscale(
                    prompt="",
                    image=tile,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    # negative_prompt = neg_prompt,
                    generator=generator,
                ).images[0]
                upscaled_img.paste(upscaled_tile, (x * upscaled_tile.size[0], y * upscaled_tile.size[1]))
        return [upscaled_img]

    else:
        return pipe_upscale(
            prompt=prompt,
            image=img,
            num_inference_steps=steps,
            guidance_scale=guidance,
            negative_prompt = neg_prompt,
            generator=generator,
        ).images

def on_mode_change(mode):
  return gr.update(visible = mode == modes['upscale4x'])
 
def on_steps_change(steps):
  global current_steps
  current_steps = steps

css = """.main-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.main-div div h1{font-weight:900;margin-bottom:7px}.main-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""
with gr.Blocks(css=css) as demo:
    gr.HTML("<h1>Testing Upscale image to X time </h1>")
    with gr.Row():
        with gr.Column(scale=55):
            with gr.Group():
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,placeholder=f"Enter prompt").style(container=False)
                    generate = gr.Button(value="Generate").style(rounded=(False, True, True, False))
                gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[2], height="auto")
        state_info = gr.Textbox(label="State", show_label=False, max_lines=2).style(container=False)
        error_output = gr.Markdown(visible=False)

        with gr.Column(scale=45):
            inf_mode = gr.Radio(label="Inference Mode", choices=list(modes.values()), value=modes['txt2img'])

            with gr.Group(visible=False) as i2i_options:
                image = gr.Image(label="Image", height=128, type="pil", tool='sketch')
                inpaint_info = gr.Markdown("Inpainting resizes and pads images to 512x512", visible=False)
                upscale_info = gr.Markdown("""Best for small images (128x128 or smaller).
                                            Bigger images will be sliced into 128x128 tiles which will be upscaled individually.
                                            This is done to avoid running out of GPU memory.""", visible=False)
                strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)
            
        with gr.Accordion("Advanced Options", open=False):
            with gr.Group():
                neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")

                n_images = gr.Slider(label="Number of images", value=1, minimum=1, maximum=4, step=1)
                with gr.Row():
                    guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
                    steps = gr.Slider(label="Steps", value=current_steps, minimum=2, maximum=100, step=1)

                with gr.Row():
                    width = gr.Slider(label="Width", value=512, minimum=64, maximum=1024, step=8)
                    height = gr.Slider(label="Height", value=512, minimum=64, maximum=1024, step=8)

                seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)
                with gr.Accordion("Memory optimization"):
                    attn_slicing = gr.Checkbox(label="Attention slicing (a bit slower, but uses less memory)", value=attn_slicing_enabled)
                    # mem_eff_attn = gr.Checkbox(label="Memory efficient attention (xformers)", value=mem_eff_attn_enabled)

    inf_mode.change(on_mode_change, inputs=[inf_mode], outputs=[i2i_options, inpaint_info, upscale_info, strength], queue=False)
    steps.change(on_steps_change, inputs=[steps], outputs=[], queue=False)
    attn_slicing.change(lambda x: switch_attention_slicing(x), inputs=[attn_slicing], queue=False)
    # mem_eff_attn.change(lambda x: switch_mem_eff_attn(x), inputs=[mem_eff_attn], queue=False)

    inputs = [inf_mode, prompt, n_images, guidance, steps, width, height, seed, image, strength, neg_prompt]
    outputs = [gallery, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

    demo.load(update_state_info, inputs=state_info, outputs=state_info, every=0.5, show_progress=False)

demo.queue()
demo.launch(debug=True, share=True, height=768)