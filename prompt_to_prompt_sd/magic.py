import inspect
import warnings
from typing import List, Optional, Union
import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn.functional as F
from config import parse_args
from huggingface_hub import HfFolder, Repository, whoami
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import PIL
from accelerate import Accelerator
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import logging
from pathlib import Path
from PIL import Image 
from torchvision import transforms
import os 
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

### ----------- Imagic StableDiffusion V0 support Xformers, Gradient checkpoint, AdamW8bit, Slice Attention --------------
class ImagicStableDiffusionPipeline(DiffusionPipeline):
    
    """
    Pipeline for imagic image editing.
    See paper here: https://arxiv.org/pdf/2210.09276.pdf

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

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
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
    
    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.
        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)
    
    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            # TODO(Patrick) - there is currently a bug with cpu offload of nn.Parameter in accelerate
            # fix by only offloading self.safety_checker for now
            cpu_offload(self.safety_checker.vision_model, device)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_xformers_memory_efficient_attention
    def enable_xformers_memory_efficient_attention(self):
        r"""
        Enable memory efficient attention as implemented in xformers.
        When this option is enabled, you should observe lower GPU memory usage and a potential speed up at inference
        time. Speed up at training time is not guaranteed.
        Warning: When Memory Efficient Attention and Sliced attention are both enabled, the Memory Efficient Attention
        is used.
        """
        self.unet.set_use_memory_efficient_attention_xformers(True)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_xformers_memory_efficient_attention
    def disable_xformers_memory_efficient_attention(self):
        r"""
        Disable memory efficient attention as implemented in xformers.
        """
        self.unet.set_use_memory_efficient_attention_xformers(False)

    def train(self, prompt: Union[str, List[str]],
            
            init_image: Union[torch.FloatTensor, PIL.Image.Image],
            height: Optional[int] = 512,
            width: Optional[int] = 512,

            generator: Optional[torch.Generator]= None, 
            embedding_learning_rate: float=0.001, 
            diffusion_model_learning_rate: float=0.001,
            text_embedding_optimization_steps: int=100,
            model_fine_tuning_optimization_steps: int= 500, 
            adam8bit_optimizer: bool= True, 
            train_text_encoder: bool= True,
            gradient_checkpointing: bool= True, 
            **kwargs
            ): 
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                'guidance_scale' is defined as 'w' of equation 2. of [Imagen paper ](https://arxiv.org/pdf/2205.11487.pdf).
                Guidance scale is enabled by setting `guidance_scale >1`. Higher guidance scale encourages to generate images that
                are closely linked to the text 'prompt' usually at the expense of lower image quality. 
            eta ('float', *optional*, defaults to 0,0)
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sample from Gaussian distribution, to be used as inputs for image generation.
                Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                the output format of the generate imgae. choose between 
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            Return_dict ('bool', *optional*, defaults to 'True'):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        accelerator= Accelerator(gradient_accumulation_steps=1, mixed_precision="fp16")
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)
        
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        ## Freeze Vae and Unet model 
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        if accelerator.is_main_process: 
            accelerator.init_trackers(
                "imagic", 
                config={
                    "embedding_learning_rate": embedding_learning_rate, 
                    "text_embedding_optimization_steps": text_embedding_optimization_steps,
                }, 
            )

        ## Get text embedding for prompt 
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0], requires_grad=True
        )
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_()
        text_embeddings_orig = text_embeddings.clone()

        ## Initialize the optimizers
        if adam8bit_optimizer: 
            optimizer = bnb.optim.AdamW8bit(
                [text_embeddings],
                lr=embedding_learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01,

            )
        else: 
            optimizer = torch.optim.Adam(
                [text_embeddings],
                lr=embedding_learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01,
            )

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()
    
        ## Configure the init image 
        if isinstance(init_image, PIL.Image.Image):
            init_image = preprocess(init_image)
        latents_dtype= text_embeddings.dtype
        init_image= init_image.to(device= self.device, dtype= latents_dtype )
        init_latent_image_dist= self.vae.encode(init_image).latent_dist 
        init_image_latents= init_latent_image_dist.sample(generator= generator)
        init_image_latents= 0.18215 * init_image_latents 

        progress_bar= tqdm (range(text_embedding_optimization_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Optimizing text embedding for prompt: {prompt}")

        global_step= 0 
        logger.info("First optimizing the text embedding to better reconstruct the init image")
        for _ in range(text_embedding_optimization_steps): 
            with accelerator.accumulate(text_embeddings): 
                # Sample noise that we will add to the latents 
                noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
                timesteps= torch.randint(1000, (1,), device= init_image_latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (This is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(init_image_latents, noise, timesteps)

                # Predict the noise residual 
                noise_pred= self.unet(noisy_latents, timesteps, text_embeddings).sample 

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)
                optimizer.step() 
                optimizer.zero_grad()
            
            ## Check if the accelerator has peformed an optimization step behind the scenes 
            if accelerator.sync_gradients: 
                progress_bar.update(1)
                global_step +=1 
            
            logs= {"loss": loss.detach().item()} #, "lr": lr_scheduler.get_last_lr()[0]}        
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step= global_step)
        
        accelerator.wait_for_everyone()

        text_embeddings.requires_grad_(False)
        
        # Now we fine tune the unet to better reconstruct the image
        self.unet.requires_grad_(True)
        self.unet.train()
        optimizer = bnb.optim.Adam8bit(
            self.unet.parameters(),  # only optimize unet
            lr=diffusion_model_learning_rate,
        )
        progress_bar = tqdm(range(model_fine_tuning_optimization_steps), disable=not accelerator.is_local_main_process)

        logger.info("Next fine tuning the entire model to better reconstruct the init image")

        for _ in range(model_fine_tuning_optimization_steps): 
            with accelerator.accumulate(self.unet.parameters()):
                # Sample noise that we'll add to the latents
                noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
                timesteps = torch.randint(1000, (1,), device=init_image_latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(init_image_latents, noise, timesteps)
                # Predict the noise residual
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

                loss= F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()
            
            ## Chec if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            logs = {"loss": loss.detach().item()}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        
        accelerator.wait_for_everyone()
        self.text_embeddings_orig = text_embeddings_orig
        self.text_embeddings = text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        alpha: float = 1.2,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        **kwargs,
    ):
        r"""
            Function invoked when calling the pipeline for generation.
            Args: all other argument similar to the previous function except for the following:
            alpha (float, ) weighting control original text_embedding 

        """


        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if self.text_embeddings is None:
            raise ValueError("Please run the pipe.train() before trying to generate an image.")
        if self.text_embeddings_orig is None:
            raise ValueError("Please run the pipe.train() before trying to generate an image.")

        text_embeddings= alpha * self.text_embeddings_orig + (1 - alpha) * self.text_embeddings

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""]
            max_length = self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        ## Get the initial random noise 
        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype

        if self.device.type == "mps":
                # randn does not exist on mps
                latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                    self.device
                )
        else:
                latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                self.device
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

        #return image, latents#StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

### ----------- Imagic StableDiffusion V2 support Xformers, Adamw8bit, 
# Configure input Arguments for pretraining and output

class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def train( args, edit_prompt, input_image=None): 
        ## Get all arguments inputs 
        #args= parse_args()
        logging_dir= Path(args.output_dir, args.logging_dir )


        accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,)

        ## set seed 
        if args.seed is not None: 
            set_seed(args.seed)
        ## Handle the repository creation 
        if accelerator.is_main_process: 
            if args.push_to_hub:
                if args.hub_model_id is None:
                    repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
                else:
                    repo_name = args.hub_model_id
                repo = Repository(args.output_dir, clone_from=repo_name)

                with open(os.path.join(args.output_dir, ".gitignore"), "w+")  as gitignore:
                    if "step_*" not in gitignore:
                        gitignore.write("step_*\n")
                    if "epoch_*" not in gitignore:
                        gitignore.write("epoch_*\n")
            
            elif args.output_dir is not None: 
                os.makedirs(args.output_dir, exist_ok=True)

        ## Get the model (Text, vae, unet, scheduler)
        if args.tokenizer_name:
            ## Use for adding the new CLIP tokenizer example (multi-Lingual model XLBERT, etc)
            tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
        elif args.pretrained_model_name_or_path:
            tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_auth_token=True)
        
        ## Load models and create wrapper for Stable D
        # Load models and create wrapper for stable diffusion
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", cache_dir=args.cache_dir, use_auth_token=True)
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", cache_dir=args.cache_dir, use_auth_token=True)
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", cache_dir=args.cache_dir, use_auth_token=True)
     
     
        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.Adam8bit
        else:
            optimizer_class = torch.optim.Adam

        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )
        
        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        ## Encode the input image 
        if input_image is  None:
            input_image= Image.open(args.input_image)

        image_transforms = transforms.Compose(        [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        init_image= image_transforms(input_image)
        init_image = init_image[None].to(device=accelerator.device, dtype=weight_dtype)
        with torch.inference_mode():
            init_latents = vae.encode(init_image).latent_dist.sample()
            init_latents = 0.18215 * init_latents

        # Encode the target text.
        text_ids = tokenizer(
            #args.target_text,
            edit_prompt,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        text_ids = text_ids.to(device=accelerator.device)
        with torch.inference_mode():
            target_embeddings = text_encoder(text_ids)[0]

        del vae, text_encoder 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        target_embeddings = target_embeddings.float()
        optimized_embeddings = target_embeddings.clone()

        # Optimize the text embeddings first.
        optimized_embeddings.requires_grad_(True)
        optimizer = optimizer_class(
            [optimized_embeddings],  # only optimize embeddings
            lr=args.emb_learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            # weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        unet, optimizer = accelerator.prepare(unet, optimizer)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("imagic", config=vars(args))

        def train_loop(pbar, optimizer, params): 
            loss_avg= AverageMeter() 
            for step in pbar: 
                with accelerator.accumulate(unet): 
                    noise= torch.randn_like(init_latents)
                    bsz= init_latents.shape[0]
                    ## Sample a random timestep for each image 
                    timesteps= torch.randint(0,  noise_scheduler.config.num_train_timesteps, (bsz,), device=init_latents.device)
                    timesteps= timesteps.long() 

                    ## Add noise to the latents according to the noise magnitude at each timestep 
                    noisy_latents= noise_scheduler.add_noise(init_latents, noise, timesteps)

                    noise_pred= unet(noisy_latents, timesteps, optimized_embeddings).sample 
                    loss= F.mse_loss(noise_pred, noise, reduction="none").mean(dim=1).mean()

                    accelerator.backward(loss)
                    # if accelerator.sync_gradients:     # results aren't good with it, may be will need more training with it.
                    #     accelerator.clip_grad_norm_(params, args.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    loss_avg.update(loss.detach_(), bsz)

                if not step % args.log_interval: 
                    logs= {"loss": loss_avg.avg.item()}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=step)
            accelerator.wait_for_everyone()

        progress_bar = tqdm(range(args.emb_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Optimizing embedding")
        train_loop(progress_bar, optimizer, optimized_embeddings)

        optimized_embeddings.requires_grad_(False)
        if accelerator.is_main_process: 
            torch.save(target_embeddings.cpu(), os.path.join(args.output_dir, "target_embeddings.pt"))
            torch.save(optimized_embeddings.cpu(), os.path.join(args.output_dir, "optimized_embeddings.pt"))
            with open(os.path.join(args.output_dir, "target_text.txt"), "w") as f:
                #f.write(args.target_text)
                f.write(edit_prompt)

        # Fine tune the diffusion model.
        optimizer = optimizer_class(
            accelerator.unwrap_model(unet).parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            # weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        optimizer = accelerator.prepare(optimizer)

        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Fine Tuning")
        unet.train()

        train_loop(progress_bar, optimizer, unet.parameters())
        
        
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                use_auth_token=True
            )
            pipeline.save_pretrained(args.output_dir)

            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

        accelerator.end_training()

class ImagicStableDiffusionV2(DiffusionPipeline):
    
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
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
    
    
    def train(self, args, edit_prompt,output_dir, input_image=None): 
        ## Get all arguments inputs 
        #args= parse_args()
        logging_dir= Path(args.output_dir, args.logging_dir )
        accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,)

        ## set seed 
        if args.seed is not None: 
            set_seed(args.seed)
        ## Handle the repository creation 
        if accelerator.is_main_process: 
            if args.push_to_hub:
                if args.hub_model_id is None:
                    repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
                else:
                    repo_name = args.hub_model_id
                repo = Repository(args.output_dir, clone_from=repo_name)

                with open(os.path.join(args.output_dir, ".gitignore"), "w+")  as gitignore:
                    if "step_*" not in gitignore:
                        gitignore.write("step_*\n")
                    if "epoch_*" not in gitignore:
                        gitignore.write("epoch_*\n")
            
            elif args.output_dir is not None: 
                os.makedirs(args.output_dir, exist_ok=True)

        ## Get the model (Text, vae, unet, scheduler)
        # if args.tokenizer_name:
        #     ## Use for adding the new CLIP tokenizer example (multi-Lingual model XLBERT, etc)
        #     tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
        # elif args.pretrained_model_name_or_path:
        #     tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_auth_token=True)
        
        ## Load models and create wrapper for Stable D
        # Load models and create wrapper for stable diffusion
        # text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", cache_dir=args.cache_dir, use_auth_token=True)
        # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", cache_dir=args.cache_dir, use_auth_token=True)
        # unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", cache_dir=args.cache_dir, use_auth_token=True)
     

        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.Adam8bit
        else:
            optimizer_class = torch.optim.Adam

        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )
        
        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.vae.to(accelerator.device, dtype=weight_dtype)
        ## Encode the input image 
        if input_image is  None:
            input_image= Image.open(args.input_image)

        image_transforms = transforms.Compose(        [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        init_image= image_transforms(input_image)
        init_image = init_image[None].to(device=accelerator.device, dtype=weight_dtype)
        with torch.inference_mode():
            init_latents = self.vae.encode(init_image).latent_dist.sample()
            init_latents = 0.18215 * init_latents

        # Encode the target text.
        text_ids = self.tokenizer(
            #args.target_text,
            edit_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        text_ids = text_ids.to(device=accelerator.device)
        with torch.inference_mode():
            target_embeddings = self.text_encoder(text_ids)[0]

        #del vae, text_encoder 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        target_embeddings = target_embeddings.float()
        optimized_embeddings = target_embeddings.clone()

        # Optimize the text embeddings first.
        optimized_embeddings.requires_grad_(True)
        optimizer = optimizer_class(
            [optimized_embeddings],  # only optimize embeddings
            lr=args.emb_learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            # weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        self.unet, optimizer = accelerator.prepare(self.unet, optimizer)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("imagic", config=vars(args))

        def train_loop(pbar, optimizer, params): 
            loss_avg= AverageMeter() 
            for step in pbar: 
                with accelerator.accumulate(self.unet): 
                    noise= torch.randn_like(init_latents)
                    bsz= init_latents.shape[0]
                    ## Sample a random timestep for each image 
                    timesteps= torch.randint(0,  noise_scheduler.config.num_train_timesteps, (bsz,), device=init_latents.device)
                    timesteps= timesteps.long() 

                    ## Add noise to the latents according to the noise magnitude at each timestep 
                    noisy_latents= noise_scheduler.add_noise(init_latents, noise, timesteps)

                    noise_pred= self.unet(noisy_latents, timesteps, optimized_embeddings).sample 
                    loss= F.mse_loss(noise_pred, noise, reduction="none").mean(dim=1).mean()

                    accelerator.backward(loss)
                    # if accelerator.sync_gradients:     # results aren't good with it, may be will need more training with it.
                    #     accelerator.clip_grad_norm_(params, args.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    loss_avg.update(loss.detach_(), bsz)

                if not step % args.log_interval: 
                    logs= {"loss": loss_avg.avg.item()}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=step)
            accelerator.wait_for_everyone()

        progress_bar = tqdm(range(args.emb_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Optimizing embedding")
        train_loop(progress_bar, optimizer, optimized_embeddings)

        optimized_embeddings.requires_grad_(False)
        # if accelerator.is_main_process: 
        #     torch.save(target_embeddings.cpu(), os.path.join(args.output_dir, "target_embeddings.pt"))
        #     torch.save(optimized_embeddings.cpu(), os.path.join(args.output_dir, "optimized_embeddings.pt"))
        #     with open(os.path.join(args.output_dir, "target_text.txt"), "w") as f:
        #         #f.write(args.target_text)
        #         f.write(edit_prompt)
        # self.target_embeddings = target_embeddings
        # self.optimized_embeddings = optimized_embeddings
        
        if accelerator.is_main_process:
            torch.save(target_embeddings.cpu(), os.path.join(output_dir, "target_embeddings.pt"))
            torch.save(optimized_embeddings.cpu(), os.path.join(output_dir, "optimized_embeddings.pt"))
            with open(os.path.join(args.output_dir, "target_text.txt"), "w") as f:
                f.write(edit_prompt)
        
        
        # Fine tune the diffusion model.
        optimizer = optimizer_class(
            accelerator.unwrap_model(self.unet).parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            # weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        optimizer = accelerator.prepare(optimizer)

        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Fine Tuning")
        self.unet.train()

        train_loop(progress_bar, optimizer, self.unet.parameters())
        
        
        # Create the pipeline using using the trained modules and save it.

        accelerator.end_training()
    
    @torch.no_grad()
    def __call__(
        self,
        text_embedding_path: str, 
        alpha: float = 0.9,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        **kwargs,
    ):
        r"""
            Function invoked when calling the pipeline for generation.
            Args: all other argument similar to the previous function except for the following:
            alpha (float, ) weighting control original text_embedding 

        """


        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        # if self.target_embeddings is None:
        #     raise ValueError("Please run the pipe.train() before trying to generate an image.")
        # if self.optimized_embeddings is None:
        #     raise ValueError("Please run the pipe.train() before trying to generate an image.")
        
        target_embeddings = torch.load(os.path.join(text_embedding_path, "target_embeddings.pt")).to("cuda")
        optimized_embeddings = torch.load(os.path.join(text_embedding_path, "optimized_embeddings.pt")).to("cuda")


        text_embeddings= alpha * target_embeddings + (1 - alpha) * optimized_embeddings

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""]
            max_length = self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        ## Get the initial random noise 
        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype

        if self.device.type == "mps":
                # randn does not exist on mps
                latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                    self.device
                )
        else:
                latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                self.device
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
