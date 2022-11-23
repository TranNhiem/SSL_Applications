from absl_mock import Mock_Flag

def read_cfg(): 
    flags= Mock_Flag() 
    base_cfg()

    return flags 

def base_cfg():
    flags = Mock_Flag() 

    flags.DEFINE_integer(
    'img_height', 512,
    'image height.')

    flags.DEFINE_integer(
    'img_width', 512,
    'image width.')

    flags.DEFINE_enum(
        "sd_model", "CompVis/stable-diffusion-v1-4", [ "CompVis/stable-diffusion-v1-4","prompthero/openjourney","runwayml/stable-diffusion-v1-5" ], 
        'The pretrained SD Model from Huggingface')
    
    flags.DEFINE_string(
        "store_path", "/data1/pretrained_weight/StableDiffusion/",
        'Path to store the download weight --> Saving in local disk')

    flags.DEFINE_integer(
    'num_diffusion_steps', 50,
    'The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.')

    flags.DEFINE_float(
    'guidance_scale', 7.5,
    'Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality')

    flags.DEFINE_boolean(
        'low_resource', False,  # 
        'Enable to True for running on 12GB GPUs ')