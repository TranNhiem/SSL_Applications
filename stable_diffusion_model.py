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

from difflib import diff_bytes
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
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import deprecate
from diffusers.configuration_utils import FrozenDict

from tqdm.auto import tqdm
import inspect
from typing import List, Tuple, Dict, Any, Optional, Union
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPModel 
from utils import image_preprocess, preprocess_mask_v2, preprocess_pil, resize_feature, set_requires_grad, prepare_mask_and_masked_image
import random
import open_clip
from torchvision import transforms
from torch.nn import functional as F
#from aitemplate.compiler import Model
import os 
import warnings

#generator = csprng.create_random_device_generator('/dev/urandom')
generator = torch.Generator(device="cuda").manual_seed(1024) #random.randint(0,10000) # change the seed to get different results

def decode_image(latents, vae,): 
    latents= 1 / 0.18215 * latents 
    image= vae.decode(latents).sample
    image= (image /2 +0.5).clamp(0,1)
    image= image.cpu().permute(0,2,3,1).numpy()
    return image 



###----------------- Stable Diffusion Inpainting update for V< 0.4  -----------------###
class StableDiffusionInpaintingPipeline_(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        #scheduler = scheduler.set_format("pt")


        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "skip_prk_steps") and scheduler.config.skip_prk_steps is False:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration"
                " `skip_prk_steps`. `skip_prk_steps` should be set to True in the configuration file. Please make"
                " sure to update the config accordingly as not setting `skip_prk_steps` in the config might lead to"
                " incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face"
                " Hub, it would be very nice if you could open a Pull request for the"
                " `scheduler/scheduler_config.json` file"
            )
            deprecate("skip_prk_steps not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["skip_prk_steps"] = True
            scheduler._internal_dict = FrozenDict(new_config)

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
        image: torch.FloatTensor,
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
        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        #print(init_image.shape)
        #init_latents = self.vae.encode(init_image.to(self.device)).sample#sample()
        image= image.to(deivce=self.device)
        init_latents = self.vae.encode(image).latent_dist.sample(generator=generator) ## .sample() --> .latent_dist.sample()

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
        image= self.vae.decode(latents).sample 
        #image = self.vae.decode(latents)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        #safety_cheker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
        #image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_cheker_input.pixel_values)

        has_nsfw_concept = 0

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}

## --------------- Stable Diffusion Inpainting Update for V > 0.6-------------------- ###
class StableDiffusionInpaintingPipeline_v2(DiffusionPipeline): 
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPFeatureExtractor,
        ):
            super().__init__()
            if hasattr(scheduler.config,"step_offset") and scheduler.config.steps_offset !=1: 
                deprecation_message = (
                    f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                    f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                    "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                    " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                    " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                    " file"
                )
                deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(scheduler.config)
                new_config["steps_offset"] = 1
                scheduler._internal_dict = FrozenDict(new_config)


###-----------------  Stable Diffusion Text2Image, Image2Image -----------------###
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
            ## Update changing information is here: 
            #https://github.com/huggingface/diffusers/releases/tag/v0.3.0
            init_latents = self.vae.encode(init_image.to(self.device)).latent_dist.sample() ## .sample() --> .latent_dist.sample()
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

###-----------------  Stable Diffusion Text2Image, Image2Image -----------------###
## Using AIT for speeding up the inference process 
class StableDiffusionPipelineAIT(DiffusionPipeline): 
    def __init__(self, 
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        work_dir: str = "/home/harry/BLIRL/SSL_Applications/tmp/", 
    ): 
        super().__init__()#vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor
        # self.register_modules(
        #     vae=vae,
        #     text_encoder=text_encoder,
        #     tokenizer=tokenizer,
        #     unet=unet,
        #     scheduler=scheduler,
        #     feature_extractor=feature_extractor,
        # )
        self.work_dir = work_dir
        self.clip_ait_exe = self.init_ait_module(model_name="CLIPTextModel")
        self.vae_ait_exe = self.init_ait_module(model_name="AutoencoderKL")
        self.unet_ait_exe = self.init_ait_module(model_name="UNet2DConditionModel")

    def init_ait_module(self, model_name): 
        mod = Model(os.path.join(self.work_dir, model_name, "test.so"))
        return mod
    
    def unet_inference(self, latent_model_input, timesteps, encoder_hidden_states): 
        exe_module= self.unet_ait_exe
        timesteps_pt=  timesteps.expand(latent_model_input.shape[0])
        inputs={
            "input0":  latent_model_input.permute((0, 2, 3, 1)) 
            .contiguous()
            .cuda()
            .half(),
            "input1": timesteps_pt.cuda().half(),
            "input2": encoder_hidden_states.cuda().half(),
        }
        ys=[] 
        num_outputs=len(exe_module.get_output_name_to_index_map())
        for i in range(num_outputs): 
            shape = exe_module.get_output_maximum_shape(i)
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode= True)
        noise_pred= ys[0].permute((0, 3, 1, 2)).float()
        return noise_pred 
    
    def clip_inference(self, input_ids, seqlen=64): 
        exe_module= self.clip_ait_exe
        bs = input_ids.shape[0]
        position_ids = torch.arange(seqlen).expand((bs, -1)).cuda()
        inputs={
            "input0": input_ids,
            "input1": position_ids,
        }
        ys = []
        num_ouputs = len(exe_module.get_output_name_to_index_map())
        for i in range(num_ouputs):
            shape = exe_module.get_output_maximum_shape(i)
            ys.append(torch.empty(shape).cuda().half())
        exe_module.run_with_tensors(inputs, ys, graph_mode=True)
        return ys[0].float()
    
    def vae_inference(self, vae_input):
            exe_module = self.vae_ait_exe
            inputs = [torch.permute(vae_input, (0, 2, 3, 1)).contiguous().cuda().half()]
            ys = []
            num_ouputs = len(exe_module.get_output_name_to_index_map())
            for i in range(num_ouputs):
                shape = exe_module.get_output_maximum_shape(i)
                ys.append(torch.empty(shape).cuda().half())
            exe_module.run_with_tensors(inputs, ys, graph_mode=True)
            vae_out = ys[0].permute((0, 3, 1, 2)).float()
            return vae_out
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )
        
        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=64,  # self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_embeddings = self.clip_inference(text_input.input_ids.to(self.device))
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
            uncond_embeddings = self.clip_inference(
                uncond_input.input_ids.to(self.device)
            )
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        # However this currently doesn't work in `mps`.
        latents_device = "cpu" if self.device.type == "mps" else self.device
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=latents_device,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )
        latents = latents.to(self.device)

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
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

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet_inference(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )

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
                ).prev_sample
            else:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae_inference(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        safety_cheker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="pt"
        ).to(self.device)
        image, has_nsfw_concept = self.safety_checker(
            images=image, clip_input=safety_cheker_input.pixel_values
        )

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
    ## Continue to Build the pipeline for speeding the inference process of the model

###----------------- Stable Diffusion Text2Video -----------------###
class StableDiffusionVideo(DiffusionPipeline): 
    pass 

###----------------- CLIP Guided Diffusion  -----------------###
class StableDiffusionCLIP_Guided(DiffusionPipeline): 
    '''
    CLIP Guided Stable Diffusion Model 
    Reference - https://github.com/Jack000/glid-3-xl

    '''
    def __init__(self, vae: AutoencoderKL, 
                        text_encoder: CLIPTextModel,
                        clip_model: CLIPModel, 
                        unet: UNet2DConditionModel, 
                        scheduler: Union[LMSDiscreteScheduler, PNDMScheduler],
                        tokenizer: CLIPTokenizer, 
                        feature_extractor: CLIPFeatureExtractor, 
                        ):

        super().__init__()
        scheduler= scheduler.set_format("pt")
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            clip_model=clip_model,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        self.mask_out = resize_feature(feature_extractor.size)
        set_requires_grad(self.text_encoder, False)
        set_requires_grad(self.clip_model, False)
    
    
    
    def spherical_dist_loss(self, x, y): 
        x= F.normalize(x, dim=-1)
        y= F.normalize(y, dim=-1)
        return (x-y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]]= "auto"): 
        if slice_size == "auto": 
            ## Half the attention head size is usually a good trade-off between speed and memory 
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

       
    def disable_attention_slicing(self):
        self.enable_attention_slicing(None)
    
    def freeze_vae(self): 
        set_requires_grad(self.vae, False)
    
    def unfreeze_vae(self):
        set_requires_grad(self.vae, True)

    def freeze_unet(self):
        set_requires_grad(self.unet, False)

    def unfreeze_unet(self):
        set_requires_grad(self.unet, True)

    @torch.enable_grad()
    def cond_fn(self, latents, timestep, index,
                        text_embeddings,
                        noise_pred_original,
                        text_embeddings_clip, 
                        clip_guidance_scale, 
                        num_cutouts, 
                        use_cutouts= True, 
                        ): 
        latents= latents.detach().requires_grad_()

        if isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
            latent_model_input = latents / ((sigma**2 + 1) ** 0.5)
        else: 
            latent_model_input = latents

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample#["sample"]

        if isinstance(self.scheduler, PNDMScheduler):
            alpha_prod_t= self.scheduler.alphas_cumprod[timestep]
            beta_prod_t= 1 - alpha_prod_t
            # compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_sample= (latents- beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            fac= torch.sqrt(beta_prod_t)
            sample= pred_original_sample *(fac) + latents *(1- fac)

        elif isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma= self.scheduler.sigmas[index]
            sample= latents- sigma* noise_pred
        else: 
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

        # decode the sample to image
        sample= 1 / 0.18215 * sample 
        image= self.vae.decode(sample).sample 
        image= (image /2 +0.5).clamp(0, 1) 

        if use_cutouts: 
            image= self.mask_out(image, num_cutouts)
        else: 
            image= transforms.Resize(self.feature_extractor.size)(image)
        
        ## Get image embedding through CLIP model 
        image_embedding_clip= self.clip_model.get_image_features(image).float() 
        image_embedding_clip= image_embedding_clip / image_embedding_clip.norm(dim=-1, keepdim=True)

        if use_cutouts: 
            dists = self.spherical_dist_loss(image_embedding_clip, text_embeddings_clip)
            dists= dists.view([num_cutouts, sample.shape[0], -1])
            loss= dists.sum(2).mean(0).sum() * clip_guidance_scale
        else:
            loss= self.spherical_dist_loss(image_embedding_clip, text_embeddings_clip).mean() * clip_guidance_scale
        
        grads= -torch.autograd.grad(loss, latents)[0]

        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents= latents.detach() + grads + (sigma**2)
            noise_pred= noise_pred_original
        else: 
            noise_pred= noise_pred_original - torch.sqrt(beta_prod_t)* grads 
        
        return noise_pred, latents 

    @torch.no_grad()
    def __call__(self, prompt: Union[str, List[str]], 
                        height: Optional[int] = 512, 
                        width: Optional[int] = 512, 
                        num_inference_steps: Optional[int]= 50, 
                        guidance_scale: Optional[float]= 7.5, 
                        clip_guidance_scale: Optional[float]= 100, 
                        clip_prompt: Optional[Union[str, List[str]]]= None, 
                        num_cutouts: Optional[int]= 4, 
                        use_cutouts: Optional[bool]=True, 
                        generator: Optional[torch.Generator]= None, 
                        latents: Optional[torch.FloatTensor]= None, 
                        output_type: Optional[str] = "pil", 
                        return_dict: bool= True, 
    ):
        
        if isinstance(prompt, str): 
            batch_size=1 
        elif isinstance(prompt, list): 
            batch_size= len(prompt)
        else: 
            raise ValueError(f"prompt type {type(prompt)} not supported")

        if height%8 !=0 or width %8 !=0: 
            raise ValueError(f"height and width must be divisible by 8, got {height} and {width}")
        
        ## Get prompt text embeddings 
        text_input= self.tokenizer(prompt, 
                                padding="max_length",
                                max_length= self.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt", 
                                )
        text_embeddings= self.text_encoder(text_input.input_ids.to(self.device))[0]

        if clip_guidance_scale > 0: 
            if clip_prompt is not None: 
                clip_text_input= self.tokenizer(clip_prompt, 
                                        padding="max_length",
                                        max_length= self.tokenizer.model_max_length,
                                        truncation=True,
                                        return_tensors="pt", 
                                        )
            else:
                clip_text_input= text_input

            text_embeddings_clip= self.clip_model.get_text_features(clip_text_input.input_ids.to(self.device)).float()
            text_embeddings_clip= text_embeddings_clip / text_embeddings_clip.norm(p=2, dim=-1, keepdim=True)

        ## "Guidance scale" is define to analog the guidance weight 'w' of equation (2) 
        do_classifier_free_guidance= guidance_scale > 1.0 
        if do_classifier_free_guidance: 
            max_length= text_input.input_ids.shape[-1]
            uncond_input= self.tokenizer([""]*batch_size, padding="max_length",
                                     max_length= max_length, 
                                     truncation=True,
                                      return_tensors="pt")

            uncond_embeddings= self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            ## Forward pass for classifier free guidance scale 
            text_embeddings= torch.cat([uncond_embeddings, text_embeddings], )
        
        ## Get the initial random noise unless the user provides it
        latent_device= "cpu" if self.device.type == "cpu" else self.device
        latents_shape= (batch_size, self.unet.in_channels, height//8, width//8) 

        if latents is None: 
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=latent_device,
            )
        
        else: 
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
        
        latents= latents.to(latent_device)
        
        ## Set timesteps 
        accepts_offset= "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {} 
        if accepts_offset: 
            extra_set_kwargs["offset"]= 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        ## if we use LMSDiscreteScheduler, the latents are multiplied with sigmas 
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents= latents* self.scheduler.sigmas[0]
        
        ## Run inference
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)): 
            # expand the latents if we are doing classifier free guidance 
            latent_model_input= torch.cat([latents] *2) if do_classifier_free_guidance else latents 
            if isinstance (self.scheduler, LMSDiscreteScheduler): 
                sigma= self.scheduler.sigmas[i]
                # The model input needs to be scaled to match the continuous ODE formulation in K-LMS 
                latent_model_input= latent_model_input / ((sigma**2 +1)**0.5)

            # Predict the noise residual 
            noise_pred= self.unet(latent_model_input, t, encoder_hidden_states= text_embeddings).sample 

            # perform classifier free guidance 
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text= noise_pred.chunk(2)
                noise_pred= noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            ## Noise 
            if clip_guidance_scale >0 : 
                text_embeddings_for_guidance= (text_embeddings.chunk(2)[1] if do_classifier_free_guidance else text_embeddings)
                ## latents computed to measure similarity with text embedding 

                noise_pred, latents= self.cond_fn(latents, t, i, 
                                                text_embeddings_for_guidance, noise_pred, 
                                                text_embeddings_clip, 
                                                clip_guidance_scale, 
                                                num_cutouts, 
                                                use_cutouts, 
                                                
                                                )
            ##Denoise process Compute the previous noisy sample x_t --> x_t -1 
            if isinstance(self.scheduler, LMSDiscreteScheduler): 
                latents= self.scheduler.step(noise_pred, i, latents).prev_sample 
            else: 
                latents= self.scheduler.step(noise_pred, t, latents).prev_sample 
            
        ## Scale and Decode the image latents with VAE 
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, None)
        has_nsfw_concept=0
        
        #return {"sample": image, "nsfw_content_detected": has_nsfw_concept}
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
    

class SD_img2img_clip_vision_encoder(): 
    pass 
