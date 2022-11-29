import os, subprocess 

import gradio as gr
from clip_interrogator import Config, Interrogator

### Extending Prompt with Artist, Flavor, Mediums, movement, Trending 
## Artist [A. B. Jackson much, A. J. Casson, A. R. Middleton Todd, A.B. Frost] Many more: https://github.com/TranNhiem/clip-interrogator/blob/main/clip_interrogator/data/artists.txt
## Movements [abstract art, abstract expressionism, abstract illusionism, academic art] Many more: https://github.com/TranNhiem/clip-interrogator/blob/main/clip_interrogator/data/movements.txt
## Trending_list = [' ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble', 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central'], ] 
## Flavor [highly detailed, sharp focus, intricate, digital painting, illustration] many more: https://github.com/TranNhiem/clip-interrogator/blob/main/clip_interrogator/data/flavors.txt
## Medium [a 3D render, a black and white photo, a bronze sculpture, a cartoon] Many more: https://github.com/TranNhiem/clip-interrogator/blob/main/clip_interrogator/data/mediums.txt



def get_cache_urls(clip_model_name="ViT-L-14/openai"):
    CACHE_URLS = [
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_artists.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_flavors.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_mediums.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_movements.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_trendings.pkl',
    ] if clip_model_name == 'ViT-L-14/openai' else [
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_artists.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_flavors.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_mediums.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_movements.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_trendings.pkl',
    ]
    return CACHE_URLS


def get_caption(image, mode, clip_model_name, best_max_flavors=32): 
 
    ## Get cache urls for each model style
    CACHE_URLS=get_cache_urls(clip_model_name)
    for url in CACHE_URLS:
        print(subprocess.run(['wget', url, '-P', 'cache'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    
    ## Configure for Caption Generation with beam search 
    config = Config()
    config.blip_num_beams = 64
    config.blip_offload = False
    config.clip_model_name = clip_model_name
    ci = Interrogator(config)

    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')

    if mode == 'best':
        return ci.interrogate(image, max_flavors=int(best_max_flavors))
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)
        