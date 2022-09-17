# TranNhiem 2022-09-02
'''
This script is used to test the performance of the stable diffusion inpainting algorithm.
Code is based on the code from HuggingFace's implementation of the stable diffusion inpainting algorithm.
https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb#scrollTo=cUBqX1sMsDR6 

# Usage: 
## 1. Scheduler Algorithm 
    + PNDM scheduler (used by default)
    + DDIM scheduler
    + K-LMS scheduler


'''

import enum
import numpy as np
import torch
#import torchcsprng as csprng
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import requests
import time 
import PIL
from PIL import Image
from io import BytesIO
from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from tqdm.auto import tqdm
import inspect
from typing import List, Tuple, Dict, Any, Optional, Union
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPModel 
from utils import image_preprocess, preprocess_mask_v2, preprocess_pil
import random
import open_clip

#generator = csprng.create_random_device_generator('/dev/urandom')
generator = torch.Generator(device="cuda").manual_seed(1024) #random.randint(0,10000) # change the seed to get different results

def decode_image(latents, vae,): 
    latents= 1 / 0.18215 * latents 
    image= vae.decode(latents)
    image= (image /2 +0.5).clamp(0,1)
    image= image.cpu().permute(0,2,3,1).numpy()
    return image 


## This class Initialization the text_2_image & image_2_image generation
class StableDiffusion_text_image_to_image_(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        #image_encoder: CLIPModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor, 
        open_clip_encoder: bool= True, 
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            #image_encoder=image_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,

        )
        #open_clip_encoder=True
        self.open_clip_encoder= open_clip_encoder
        if open_clip_encoder:
            # Other models is here: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/pretrained.py
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            self.preprocess_img=preprocess
            self.image_encoder_open_clip= model
        
        ## Using image_coder from OpenAI_Clip 
        else: 
            self.image_preprocess= None#CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch14")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        mode: str = "prompt", 
        using_clip: bool= False,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        init_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        strength: Optional[float] = 0.8,
        return_intermediate: bool = False,
        output_type: Optional[str] = "pil",
    
        **kwargs,
    ):
        start_time= time.time()
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"prompt must be a string or a list of strings but U provided {type(prompt)}")

        if height % 8!=0 or width%8!=0: 
            raise ValueError(f"image height and width have to be divisible by 8 but you provide {height} & {width}")
        
        # Setting the timesteps
        accepts_offset = "offset" in set(inspect.signature(
            self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        if mode =="prompt": 
            ## Checking the size image generation

            # Get the initial random noise 
            latents= torch.randn(
                (batch_size, self.unet.in_channels, height//8, width //8), 
                generator = generator, 
                 device=self.device,
            )
            t_start = 0 
        
        elif mode == "image": 
            if not init_image: 
                raise ValueError("you are using image_to_image please provide the init_image")   
            if strength <0 or strength > 1: 
                raise ValueError(f"The value of strenth should in [0.0, 1.0] but is {strength}")

            if not isinstance(init_image, torch.FloatTensor): 
                if using_clip: 
                    print(f'img2img with semantic embedding feature extractor')
                    if not isinstance(init_image, torch.FloatTensor): 
                        if self.open_clip_encoder: 
                            init_image= self.preprocess(init_image).unsqueeze(0).to(self.device) 
                            image_embedding= self.image_encoder_open_clip.encode_image(init_image)
                        else: 
                            init_image= self.image_preprocess(image= init_image, return_tensors="pt").to(self.device)
                            image_embedding= self.image_encoder.get_image_features(**init_image)
                    else: 
                        if self.open_clip_encoder: 
                            image_embedding= self.image_encoder_open_clip.encode_image(init_image)
                        else:
                            image_embedding= self.image_encoder.get_image_features(**init_image)
                    image_embedding=image_embedding.unsqueeze(1)
                else:
                    print(f' img2img with text condition')
                    init_image= image_preprocess(init_image, full_resolution=True)
                    # Encode the init image into latents and scale the latents
                    init_latents = self.vae.encode(init_image.to(self.device)).sample()
                    init_latents = 0.18215 * init_latents
                    # prepare init_latents noise to latents
                    init_latents = torch.cat([init_latents] * batch_size)

                    # Get the original timestep using init_timestep
                    init_timestep = int(num_inference_steps*strength) + offset
                    init_timestep = min(init_timestep, num_inference_steps)
                    timesteps = self.scheduler.timesteps[-init_timestep]
                    timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)

                    # Adding noise to to the latents for each timesteps
                    noise = torch.randn(init_latents.shape,
                                        generator=generator, device=self.device)
                    init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)
                    latents= init_latents
                    t_start = max(num_inference_steps - init_timestep + offset, 0)
                  
        
        # Getting the text prompts embedding from frozen CLIP Text encoder
        ## Still getting the text condition Input for the Variant image 
        print(f"This is prompt input: {prompt}")
        text_input = self.tokenizer(prompt,
                                    padding="max_length",
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt",
                                    )
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device))[0]

        # Guidance_scale is defined analog to the guidance weight more information is available in the paper
        # Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`

        do_classifier_free_guidance = guidance_scale > 1.0
        if using_clip:
            if do_classifier_free_guidance:
                uncond_embeddings= torch.zeros_like(image_embedding)
                ##Avoiding forward pass 2 times 
                image_embedding=torch.cat([uncond_embeddings, image_embedding])
            # Get the initial random noise 
            latents_shape= (batch_size, self.unet.int_channels, height //8, width //8)
            if latents is None: 
                latents= torch.randn(
                    latents_shape, 
                    generator= generator, 
                    device= self.device, 
                )
            else: 
                if latents.shape != latents_shape: 
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
                latents= latents.to(self.device)
                
        else: 
            #Get unconditional embedding for classifier free guidance
            if do_classifier_free_guidance:
                max_length = text_input.input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )# truncation=True,
                uncond_embeddings = self.text_encoder(
                    uncond_input.input_ids.to(self.device))[0]

                # Concatenate the text embeddings and uncond embeddings into single batch to avoid doing 2 forward passes
                text_embeddings = torch.cat(
                    [text_embeddings, uncond_embeddings],)  # (2* batch_size, 512)

        # Different Schedulers have different signatures parameters
        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):latents = latents * self.scheduler.sigmas[0]

        # DDIMS scheduler has `eta [0-> 1]` parameter  DDIM paper: https://arxiv.org/abs/2010.02502
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        
        
        if using_clip: 
            for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=image_embeddings)["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]
   
            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents)

            image = (image["sample"] / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            has_nsfw_concept = 0
            if output_type == "pil":
                image = self.numpy_to_pil(image)

            return {"sample": image, "nsfw_content_detected": has_nsfw_concept}

        else: 
            # total= len(self.scheduler.timesteps[t_start:])):
            intermediate_images=[] 
            for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:]), ):
                # expand the latents if doing classifier free guidance
                latent_mdoel_input = torch.cat(
                    [latents]*2) if do_classifier_free_guidance else latents

                if isinstance(self.scheduler, LMSDiscreteScheduler): 
                    sigma= self.scheduler.sigmas[i]
                    latent_model_input= latent_model_input / ((sigma**2 +1)**0.5)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_mdoel_input, t, encoder_hidden_states=text_embeddings)['sample']

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)
                
                # Compute the previous noisy smaple x_t --> x_t -1
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs
                    )["prev_sample"]
                else: 
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]
                if return_intermediate: 
                    decoded_image= decode_image(latents, self.vae)
                    intermediate_images.append(self.numpy_to_pil(decoded_image))


            # scale and decode the image latents with vae
            has_nsfw_concept = None

            if return_intermediate:
                image = intermediate_images[-1]
            else:
                image = decode_image(latents, self.vae)
                safety_cheker_input = self.feature_extractor(
                    self.numpy_to_pil(image), return_tensors="pt"
                ).to(self.device)
                image, has_nsfw_concept = self.safety_checker(
                    images=np.asarray(image), clip_input=safety_cheker_input.pixel_values
                )
                image = self.numpy_to_pil(image)

            return {
                "sample": image,
                "nsfw_content_detected": has_nsfw_concept,
                "intermediates": intermediate_images,
                "time": time.time() - start_time,
            }

class StableDiffusionInpaintingPipeline_(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        init_image: torch.FloatTensor,
        mask_image: torch.FloatTensor = None,
        strength: float = 0.75,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        inpainting_v2: bool =False, 

    ):

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # encode the init image into latents and scale the latents

        #print(init_image.shape)
        init_latents = self.vae.encode(init_image.to(self.device)).sample()
        init_latents = 0.18215 * init_latents
        init_latents_orig = init_latents

        # preprocess mask
        if mask_image is not None: 
            if inpainting_v2: 
                # preprocess mask
                mask = preprocess_mask_v2(mask_image).to(self.device)
                #mask = torch.cat([mask] * batch_size)
                #check sizes
                if not mask.shape == init_latents.shape:
                    raise ValueError(f"The mask and init_image should be the same size!")
            else: 
                mask = mask_image.to(self.device)
        

        # prepare init_latents noise to latents
        init_latents = torch.cat([init_latents] * batch_size)

        # get the original timestep using init_timestep
        init_timestep = int(num_inference_steps*strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)


        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
        #for i, t in enumerate(self.scheduler.timesteps[t_start:]):
        
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(
                    noise_pred, i, latents, **extra_step_kwargs
                )["prev_sample"]
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]
            
            if mask_image is not None: 
                #masking
                if t > 1:
                    t_noise = torch.randn(latents.shape, generator=generator, device=self.device)
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, t_noise, t-1)
                    latents = init_latents_proper * mask    +    latents * (1-mask)
                else:
                    latents = init_latents_orig * mask    +    latents * (1-mask)



        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        #safety_cheker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
        #image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_cheker_input.pixel_values)

        has_nsfw_concept = 0

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}

class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: CLIPFeatureExtractor,
        safety_checker: StableDiffusionSafetyChecker,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        mode: str = "prompt",
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        strength: float = 0.8,
        init_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        return_intermediates: bool = False,
        **kwargs,
    ):
        start_time = time.time()
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        if mode == "prompt":
            if height % 8 != 0 or width % 8 != 0: ###image conditioning, should be (1, 4, 64, 64)
                raise ValueError(
                    f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
                )
            # get the intial random noise
            latents = torch.randn(
                (batch_size, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
                device=self.device,
            )
            t_start = 0
        elif mode == "image":
            if not init_image:
                raise ValueError(
                    "If `mode` is 'image' you have to provide an `init_image`."
                )
            if strength < 0 or strength > 1:
                raise ValueError(
                    f"The value of strength should in [0.0, 1.0] but is {strength}"
                )
            if not isinstance(init_image, torch.FloatTensor):
                init_image = preprocess_pil(init_image)
            # print(init_image.shape, generator, self.device)
            # encode the init image into latents and scale the latents
            init_latents = self.vae.encode(init_image.to(self.device)).sample()
            init_latents = 0.18215 * init_latents

            # expand init_latents for batch_size
            init_latents = torch.cat([init_latents] * batch_size)

            # get the original timestep using init_timestep
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            timesteps = self.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor(
                [timesteps] * batch_size, dtype=torch.long, device=self.device
            )

            # add noise to latents using the timesteps
            noise = torch.randn(
                init_latents.shape, generator=generator, device=self.device
            )
            init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)
            latents = init_latents
            t_start = max(num_inference_steps - init_timestep + offset, 0)

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        intermediate_images = []
        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(
                    noise_pred, i, latents, **extra_step_kwargs
                )["prev_sample"]
            else:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )["prev_sample"]
            if return_intermediates:
                decoded_image = decode_image(latents, self.vae)
                intermediate_images.append(self.numpy_to_pil(decoded_image))

        # scale and decode the image latents with vae
        has_nsfw_concept = None
        if return_intermediates:
            image = intermediate_images[-1]
        else:
            image = decode_image(latents, self.vae)
            # safety_cheker_input = self.feature_extractor(
            #     self.numpy_to_pil(image), return_tensors="pt"
            # ).to(self.device)
            # image, has_nsfw_concept = self.safety_checker(
            #     images=np.asarray(image), clip_input=safety_cheker_input.pixel_values
            # )
            image = self.numpy_to_pil(image)

        return {
            "sample": image,
            "nsfw_content_detected": has_nsfw_concept,
            "intermediates": intermediate_images,
            "time": time.time() - start_time,
        }
class StableDiffusionVideo(DiffusionPipeline): 
    pass 

class SD_img2img_clip_vision_encoder(): 
    pass 
