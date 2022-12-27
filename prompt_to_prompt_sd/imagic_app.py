import gradio as gr
from PIL import Image
from io import BytesIO
import torch
import os
from config import read_cfg, parse_args
from magic import train 
from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusionPipeline
from magic import ImagicStableDiffusionPipeline, ImagicStableDiffusionV2
has_cuda = torch.cuda.is_available()
flag = read_cfg()
FLAGS = flag.FLAGS
#Init diffusion model
SD_Model = FLAGS.sd_model
store_path= FLAGS.store_path
#os.system("pip install git+https://github.com/fffiloni/diffusers")
device = "cuda" 


## Create the Finetune Pipeline for the Imagic Realworld image Editing


generator = torch.Generator("cuda").manual_seed(0)

###--------------Version 2-------------------------
## Original Design 
#step 1 --> Train (finetune on init image) --> Save Model  --> Load the Model Back
## Editting Design 
## Step 1 --> Train (finetune on init image) --> Run model in the inference mode

def infer(prompt, init_image, trn_steps):
    init_image = Image.open(init_image).convert("RGB")
    init_image = init_image.resize((512, 512))
    args = parse_args()
    # updating argparse arguments
    args.cache_dir = store_path
    args.pretrained_model_name_or_path= "runwayml/stable-diffusion-v1-5"
    #args.target_text= prompt
    args.emb_train_steps=500 
    args.max_train_steps= trn_steps

    ### Using this for Verison 1 
#     ###--------------Version 1-------------------------
#     imagic_pipe = ImagicStableDiffusionPipeline.from_pretrained(
#         #"CompVis/stable-diffusion-v1-4",
#         "runwayml/stable-diffusion-v1-5",
#         safety_checker=None,
#         #custom_pipeline=ImagicStableDiffusionPipeline,
#         cache_dir=store_path,
#         scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
#     ).to(device)

#     res = imagic_pipe.train(
#         prompt,
#         init_image,
#         guidance_scale=7.5,
#         num_inference_steps=50,
#         generator=generator,
#         text_embedding_optimization_steps=500,
#         model_fine_tuning_optimization_steps=trn_steps)
    
#    # with torch.no_grad():
#    #     torch.cuda.empty_cache()
#     image = imagic_pipe(alpha=1).images


    ## Training and save model 
    # train(args,edit_prompt=prompt, input_image= init_image)
    # model_path = args.output_dir 
    # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    # pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    # target_embeddings = torch.load(os.path.join(model_path, "target_embeddings.pt")).to("cuda")
    # optimized_embeddings = torch.load(os.path.join(model_path, "optimized_embeddings.pt")).to("cuda")
    # g_cuda = torch.Generator(device='cuda')
    # seed = 4324 #@param {type:"number"}
    # g_cuda.manual_seed(seed)
    # alpha = 0.9 #@param {type:"number"}
    # num_samples = 4 #@param {type:"number"}
    # guidance_scale = 3 #@param {type:"number"}
    # num_inference_steps = 50 #@param {type:"number"}
    # height = 512 #@param {type:"number"}
    # width = 512 #@param {type:"number"}
    # edit_embeddings = alpha*target_embeddings + (1-alpha)*optimized_embeddings


    # with torch.autocast("cuda"), torch.inference_mode():
    #     images = pipe(
    #         text_embeddings=edit_embeddings,
    #         height=height,
    #         width=width,
    #         num_images_per_prompt=num_samples,
    #         num_inference_steps=num_inference_steps,
    #         guidance_scale=guidance_scale,
    #         generator=g_cuda
    #     ).images

    ### Using this for Verison 1 
    ###--------------Version 1-------------------------

    imagic_pipe = ImagicStableDiffusionV2.from_pretrained(
        #"CompVis/stable-diffusion-v1-4",
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
        #custom_pipeline=ImagicStableDiffusionPipeline,
        cache_dir=store_path,
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    ).to(device)
    
    imagic_pipe.train(args,edit_prompt=prompt,output_dir=args.output_dir, input_image= init_image)
     
    images_2 = imagic_pipe(text_embedding_path=args.output_dir, alpha=1.2).images
    
    # with torch.autocast("cuda"), torch.inference_mode():
    #    images = imagic_pipe(alpha=0.7).images
    
    # return images, images_2
    return images_2
   

title = """
    <div style="text-align: center; max-width: 650px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-top: 7px;">
            Imagic Stable Diffusion • Community Pipeline
        </h1>
        </div>
         <p style="margin-top: 10px; font-size: 94%">
        Text-Based Real Image Editing with Diffusion Models
        <br />This pipeline aims to implement <a href="https://arxiv.org/abs/2210.09276" target="_blank">this paper</a> to Stable Diffusion, allowing for real-world image editing.
        
        </p>
        <br /><img src="https://user-images.githubusercontent.com/788417/196388568-4ee45edd-e990-452c-899f-c25af32939be.png" style="margin:7px 0 20px;"/>
       
        <p style="font-size: 94%">
            You can skip the queue by duplicating this space or run the Colab version: 
            <span style="display: flex;align-items: center;justify-content: center;height: 30px;">
            <a href="https://huggingface.co/spaces/fffiloni/imagic-stable-diffusion?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a>       
            </span>
        </p>
    </div>
"""

article = """
    <div class="footer">
        <p><a href="https://github.com/huggingface/diffusers/tree/main/examples/community#imagic-stable-diffusion" target="_blank">Community pipeline</a> 
        baked by <a href="https://github.com/MarkRich" style="text-decoration: underline;" target="_blank">Mark Rich</a> - 
        Gradio Demo by 🤗 <a href="https://twitter.com/fffiloni" target="_blank">Sylvain Filoni</a>
        </p>
    </div>
"""

css = '''
    #col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
    a {text-decoration-line: underline; font-weight: 600;}
    .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
'''


with gr.Blocks(css=css) as block:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)

        prompt_input = gr.Textbox(label="Target text", placeholder="Describe the image with what you want to change about the subject")
        image_init = gr.Image(source="upload", type="filepath",label="Input Image")
        trn_steps = gr.Slider(250, 1000, value=500, label="finetuning steps")
        submit_btn = gr.Button("Train")
        
        #image_output = gr.Image(label="Edited image")
        image_output= gr.Gallery(label="Edited images",show_label=True).style(grid=[2], height="auto").style(height=400)
        # examples=[['a sitting dog','imagic-dog.png', 250], ['a photo of a bird spreading wings','imagic-bird.png',250]]
        # ex = gr.Examples(examples=examples, fn=infer, inputs=[prompt_input,image_init,trn_steps], outputs=[image_output], cache_examples=False, run_on_click=False)
        
        
        gr.HTML(article)

    submit_btn.click(fn=infer, inputs=[prompt_input,image_init,trn_steps], outputs=[image_output])
    
block.queue(max_size=12).launch(show_api=False)