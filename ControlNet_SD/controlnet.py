'''
TranNhiem 2023/03/02 

    Support Features: 
        + 

    Expected Product for Service: 
        + Kids Education App 
        + Increasing Art Creation 
        + Home Design and Architecture Company for Home Design adding New Product 


Reference: 
    https://github.com/lllyasviel/ControlNet/blob/main/docs/annotator.md 
    https://huggingface.co/spaces/hysts/ControlNet 
    ## Ultra Fast ControlNet setup 
    https://huggingface.co/blog/controlnet

    ## Install Condition Models 
    https://github.com/takuma104/controlnet_hinter 

'''



import os 
import cv2
from PIL import Image
import numpy as np
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

from diffusers import UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector
import controlnet_hinter

##---------------------- Availalbe ControlNet Condition Model ----------------------##

'''
lllyasviel/sd-controlnet-depth
lllyasviel/sd-controlnet-hed
lllyasviel/sd-controlnet-normal
lllyasviel/sd-controlnet-scribble
lllyasviel/sd-controlnet-seg
lllyasviel/sd-controlnet-openpose
lllyasviel/sd-controlnet-mlsd

'''

test_image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")

test_image

weight_path= "/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weight/StableDiffusion/"
## check the weight path if not create the path
if not os.path.exists(weight_path):
    os.makedirs(weight_path)


#****************************************************************************************#
##---------------------- ControlNet with Canny Edege Condition ----------------------##
#****************************************************************************************#

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",  cache_dir=weight_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", cache_dir=weight_path, controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

image = np.array(test_image)
low_threshold = 100
high_threshold = 200
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
canny_image

prompt = ", best quality, extremely detailed"
prompt = [t + prompt for t in [ "taylor swift"]] #"Sandra Oh", "Kim Kardashian", "rihanna",
generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]

# output = pipe(
#     prompt,
#     canny_image,
#     negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] ,
#     num_inference_steps=20,
#     generator=generator,
# )

# image=output[0][0]
# image.save("./canny_output.png")

#****************************************************************************************#
##---------------------- ControlNet with HED Edege Condition ----------------------##
#****************************************************************************************#

#hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")
#hed_img= hed(test_image)
hed_img= controlnet_hinter.hint_hed(test_image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed",  cache_dir=weight_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", cache_dir=weight_path, controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()
# output = pipe(
#     prompt,
#     hed_img,
#     negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] ,
#     num_inference_steps=20,
#     generator=generator,
# )

# image=output[0][0]
# image.save("./hed_1_output.png")


#****************************************************************************************#
##---------------------- ControlNet with scribble Edege Condition ----------------------##
#****************************************************************************************#
scribble_img= controlnet_hinter.hint_scribble(test_image)
scribble_img.save("./scribble.png")
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble",  cache_dir=weight_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", cache_dir=weight_path, controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

# output = pipe(
#     prompt,
#     scribble_img,
#     negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] ,
#     num_inference_steps=20,
#     generator=generator,
# )

# image=output[0][0]
# image.save("./scribble_output.png")


#****************************************************************************************#
##---------------------- ControlNet with Hough & MLSD  Edege Condition ----------------------##
#****************************************************************************************#
# mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
# mlsd_1= mlsd(test_image)
# mlsd_1.save("./mlsd_1.png")

mlsd_2= controlnet_hinter.hint_hough(test_image)
mlsd_2.save("./mlsd_2.png")

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd",  cache_dir=weight_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", cache_dir=weight_path, controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()
# output = pipe(
#     prompt,
#     mlsd_2,
#     negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] ,
#     num_inference_steps=20,
#     generator=generator,
# )

# image=output[0][0]
# image.save("./mlsd_1_output.png")


#****************************************************************************************#
##---------------------- ControlNet with Depth and Normal Map Condition ----------------------##
#****************************************************************************************#
depth= controlnet_hinter.hint_depth(test_image)
depth.save("./depth_img.png")

normal_map= controlnet_hinter.hint_normal(test_image)
normal_map.save("./normal_map.png")

controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth",  cache_dir=weight_path, torch_dtype=torch.float16)
controlnet_normal = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal",  cache_dir=weight_path, torch_dtype=torch.float16)
control_depth_normal= [ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth",  cache_dir=weight_path, torch_dtype=torch.float16), 
                       ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal",  cache_dir=weight_path, torch_dtype=torch.float16)]
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", cache_dir=weight_path, controlnet=controlnet_depth, torch_dtype=torch.float16)
pipe_1 = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", cache_dir=weight_path, controlnet=controlnet_normal, torch_dtype=torch.float16)
pipe_2= StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", cache_dir=weight_path, controlnet=control_depth_normal, torch_dtype=torch.float16)
## For Depth image Condition 
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()
# pipe.enable_xformers_memory_efficient_attention()
# output = pipe(
#     prompt,
#     depth,
#     negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] ,
#     num_inference_steps=20,
#     generator=generator,
# )

# image=output[0][0]
# image.save("./depth_output.png")

# ## For Normal Map Condition
# pipe_1.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe_1.enable_model_cpu_offload()
# pipe_1.enable_xformers_memory_efficient_attention()
# output = pipe_1(
#     prompt,
#     normal_map,
#     negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] ,
#     num_inference_steps=20,
#     generator=generator,
# )

# image=output[0][0]
# image.save("./normal_map_output.png")

# ## For Depth && Normal Map Condition
# pipe_2.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe_2.enable_model_cpu_offload()
# pipe_2.enable_xformers_memory_efficient_attention()
# img_depth_normal= [depth, normal_map]

# output = pipe_2(
#     prompt,
#     img_depth_normal,
#     negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] ,
#     num_inference_steps=20,
#     generator=generator,
# )
# image=output[0][0]
# image.save("./depth_normal_output.png")

#****************************************************************************************#
##---------------------- ControlNet with Segmentation Map Condition ----------------------##
#****************************************************************************************#
# segmentation_map= controlnet_hinter.hint_segmentation(test_image)
# segmentation_map.save("./segmentation_map.png")

# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg",  cache_dir=weight_path, torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", cache_dir=weight_path, controlnet=controlnet, torch_dtype=torch.float16
# )
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()
# pipe.enable_xformers_memory_efficient_attention()
# output = pipe(
#     prompt,
#     segmentation_map,
#     negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] ,
#     num_inference_steps=20,
#     generator=generator,
# )

# image=output[0][0]
# image.save("./segmentation_output.png")
