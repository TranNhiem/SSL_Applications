
import torch
import os
import sys
import gradio as gr
import numpy as np
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from OFA.tasks.mm_tasks.refcoco import RefcocoTask
from OFA.models.ofa import OFAModel
from torchvision import transforms
import torchvision
from matplotlib import pyplot as plt
from PIL import Image,  ImageFont, ImageDraw
import PIL
import cv2
import random
from utils import bin2coord, coord2bin, decode_fn, preprocess_image_v2, preprocess_mask_v2
from stable_diffusion_model import StableDiffusionInpaintingPipeline_

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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# ---------------------------------------------------------
# Section for Visual Grounding Language - Visual Model
# ---------------------------------------------------------

use_fp16 = True
tasks.register_task("refcoco", RefcocoTask)

# Specify some option for evaluation
parser = options.get_generation_parser()
# Patch-image-size =384?, checking Patch_size=16 in configuration.
input_args = ["", "--task=refcoco", "--beam=10", "--path=/home/rick/pretrained_weight/OFA/ofa_large.pt", "--bpe-dir=./OFA/utils/BPE",
              "--no-repeat-ngram-size=3", "--patch-image-size=384"]
args = options.parse_args_and_arch(parser, input_args)
cfg = convert_namespace_to_omegaconf(args)
task = tasks.setup_task(cfg.task)
# Checking out some of configuration
print(task.cfg.patch_image_size)

# Loading model
models, cfg = checkpoint_utils.load_model_ensemble(
    utils.split_paths(cfg.common_eval.path), task=task)
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if torch.cuda.is_available() and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)
    torch.cuda.empty_cache()

# Initialize Generator
generator = task.build_generator(models, cfg.generation)

# Transform the input image to tensor processing by model
# Standard normalization
# mean= [0.5, 0.5, 0.5]
# std= [0.5, 0.5, 0.5]
# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
patch_resize_transform = transforms.Compose([lambda image: image.convert("RGB"),
                                            transforms.Resize(
                                                (task.cfg.patch_image_size, task.cfg.patch_image_size), interpolation=Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                mean=mean, std=std),
                                             ])
# Text proprocessing
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def encode_text(text, length=None, append_bos=False, append_eos=False, ):
    line = [
        task.bpe.encode(' {}'.format(word.strip()))
        if not word.startswith('<code_') and not word.startswith('<bin_') else word
        for word in text.strip().split()]

    line = ' '.join(line)
    s = task.tgt_dict.encode_line(
        line=line,
        add_if_not_exist=False,
        append_eos=False,).long()

    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

def construct_sample(image: Image, instruction: str):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    instruction = encode_text(' {}'.format(
        instruction.lower().strip()), append_bos=True, append_eos=True).unsqueeze(0)
    instruction_length = torch.LongTensor(
        [s.ne(pad_idx).long().sum() for s in instruction])
    sample = {
        "id": np.array(['42']),
        "net_input": {
            "src_tokens": instruction,
            "src_lengths": instruction_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        }
    }
    return sample

def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

# Getting the Coordinate from the
def get_ofa_visual_grounding(image: Image, question):
    # construct instruction
    w, h = image.size
    instruction = 'which region is the text " ' + question + ' " describe?'

    # Construct input sample
    sample = construct_sample(image, instruction)
    sample = utils.move_to_cuda(
        sample) if torch.cuda.is_available() else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    # Generate result
    with torch.no_grad():
        hypos = task.inference_step(generator, models, sample)
        tokens, bins, imgs = decode_fn(
            hypos[0][0]["tokens"], task.tgt_dict, task.bpe, generator)
        torch.cuda.empty_cache()

    # Display result
    w_resize_ratio = task.cfg.patch_image_size / w
    h_resize_ratio = task.cfg.patch_image_size / h
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    coord_list = bin2coord(bins, w_resize_ratio, h_resize_ratio, task)
    coords = [int(coord_list[0]), int(coord_list[1]),
              int(coord_list[2]), int(coord_list[3])]

    return coords, img

# Creating the mask for given Coordinate and Image size
def creat_mask(coords, img):
    temp_mask_image = np.zeros(img.shape, dtype="uint8")
    mask_image = PIL.Image.fromarray(np.uint8(temp_mask_image)).convert("RGB")
    mask_image.paste((255, 255, 255), coords)
    #mask_image = np.array(mask_image)
    return mask_image


# ---------------------------------------------------------
# Section for Stable Diffusion Model
# ---------------------------------------------------------
#generator = csprng.create_random_device_generator('/dev/urandom')
pipeimg = StableDiffusionInpaintingPipeline_.from_pretrained(
    "CompVis/stable-diffusion-v1-4", revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
).to("cuda")

def dummy(images, **kwargs): return images, False
pipeimg.safety_checker = dummy

def app_run(init_image, question, prompt, samples_num, step_num, scale):

    # Init image
    pil_image = transforms.ToPILImage()
    init_image = pil_image(init_image)
    w, h = init_image.size
    init_image = init_image.resize((512, 512))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    init_img = transform(init_image)
    print(f"init_image shape {init_img.shape}")

    # Creat the image return with bounding Box
    # .type(torch.uint8), normalize=True, scale_each=True,
    init_image_ = torchvision.utils.make_grid(
        init_img, normalize=True, scale_each=True,)
    init_image_ = Image.fromarray(init_image_.mul(255).add_(0.5).clamp_(
        0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    # Currently image return will in size 512x512
    image_bbox = Image.new('RGB', (512, 512))

    # mask_image
    coords, img = get_ofa_visual_grounding(init_image, question)
    init_image = preprocess_image_v2(init_image)
    mask_image = creat_mask(coords, img)

    # adding coordinate to Original image + text tile
    draw_ = ImageDraw.Draw(init_image_, "RGBA")
    # use a bitmap font
    font = ImageFont.truetype("/home/rick/code_spaces/SSL_Applications/text_font/StonyIslandNF.ttf", 24)
    draw_.text((200, 0), str(question), (255, 255, 255),font=font)
    draw_.rectangle(coords, fill=(200, 100, 0, 127))
    image_bbox.paste(init_image_)
    image_bbox.resize((w, h), resample=PIL.Image.LANCZOS)  # (w//8, h//8
    # Seed Random for generating image
    generator = torch.Generator(device="cuda").manual_seed(
        random.randint(0, 10000))  # change the seed to get different results
    with torch.cuda.amp.autocast():
        images = pipeimg(prompt=[prompt]*samples_num,
                         init_image=init_image,
                         mask_image=mask_image,
                         # strength= strength,
                         num_inference_steps=step_num,
                         guidance_scale=scale,
                         generator=generator,
                         inpainting_v2=True,
                         )["sample"]

    return images, [image_bbox]  # [image_bbox] for output is gr.Gallary


block = gr.Blocks(css=".container { max-width: 1300px; margin: auto; }")


examples = [['./Bird_images/343785.jpg', 'yellow Bird', 'Butterfly sitting on the tree branch.'],
            ['./Bird_images/beagle_24.jpg', 'dog', 'a Hokkaido Ken dog is running in the yellow flower garden.'],
            ['./Bird_images/Bombay_33.jpg', 'black cat', 'The Somali cat is lying on the blue sofa.'],
            ['./Bird_images/343802.jpg', 'Bird', 'A beautiful eagle sitting on the tree branch.'],
            ['./Bird_images/343787.jpg', 'Bird','blue angery bird sitting on the tree branch.'],
            ]

with block as demo:
    gr.Markdown(
        "<h1><center> Image Inpainting App  🎨 - 🖼️  </center></h1>")

    gr.Markdown(
        "<h2><center> Power by Visual Grounding with Natural Language Understand & Text-to-Image Model </center></h2> ")

    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                with gr.Row().style(mobile_collapse=False, equal_height=True):
                    with gr.Column(scale=1, min_width=900):
                        txt_ofa_ques = gr.Textbox(label=" 1st: 📜 Describe the REGION expected to CHANGE in your picture 🔥.",
                                                  placeholder="Enter text describe to describe the expected removing object here...").style(
                            border=(True, False, True, True),
                            rounded=(True, False, False, True), container=False,)
                    
                    with gr.Column(scale=2, min_width=100):
                        btn = gr.Button("3rd: Run").style(
                            margin=False, rounded=(True, True, True, True),)

            with gr.Row().style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(label="2nd: 📜 Inpainting 🎨 with Your prompt", placeholder="Enter In-painting expected object here...", show_label=True, max_lines=1).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True), container=False,)

        with gr.Row().style(mobile_collapse=False, equal_height=True):
            samples_num = gr.Slider(label="Number of Generated Image",
                                    minimum=1, maximum=10, value=2, step=1,)  # show_label=False
            steps_num = gr.Slider(
                label="Generatio of steps", minimum=2, maximum=499, value=80, step=1,)  # show_label=False
            scale = gr.Slider(label="Guidance scale", minimum=0.0,
                              maximum=30, value=7.5, step=0.1,)  # show_label=False

        # with gr.Row().style(mobile_collapse=False, equal_height=True):
        #     option = gr.Radio(label="Inpainting Area", default="Object Region Area", choices=["Object Region Area"
        #                 "Background Area"], show_label=True)

        with gr.Row().style(mobile_collapse=False, equal_height=True):  # equal_height=True

            with gr.Column(scale=1, min_width=600):
                image = gr.Image(
                    source="upload", label="Input image", type="numpy")

            with gr.Column(scale=2, min_width=400, min_height=400):
                image_box = gr.Gallery(label='Expected Replacement Object').style(grid=[2], height="auto",)  # container=True
                #image_box = gr.Image(label='Expected Replacement Object')

        gallery = gr.Gallery(label="Generated Inpainting Images 🔥 🖼️",
                             show_label=True).style(grid=[2], height="auto")

        gr.Markdown("## Examples Image 🖼️  with text 📜 desciptions.")
        ex = gr.Examples(examples=examples, fn=app_run, inputs=[
                         image, txt_ofa_ques, text, samples_num, steps_num, scale], outputs=[gallery, image_box], cache_examples=False)
        #ex.dataset.headers = [""]
        text.submit(fn=app_run, inputs=[
                    image, txt_ofa_ques, text, samples_num, steps_num, scale], outputs=[gallery, image_box])

        btn.click(fn=app_run, inputs=[
                  image, txt_ofa_ques, text, samples_num, steps_num, scale], outputs=[gallery, image_box])

demo.launch( server_name="140.115.53.102", server_port=4444,
            share=True, enable_queue=True)  # enable_queue=True, # show_error=True, debug=True,

