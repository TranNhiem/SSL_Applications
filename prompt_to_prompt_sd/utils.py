import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import PIL
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
from diffusers import LMSDiscreteScheduler, DDPMScheduler
import random 
# import torchvision.transforms as T

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)

def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents

def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent

@torch.no_grad()
def text2image_ldm_stable(
    model,
    controller,
    prompt: List[str],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    batch_size = len(prompt)
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)

    width=512
    height=512
    if latent is not None:
        width = latent.shape[-1] * 8
        height = latent.shape[-2] * 8
        
    width = width - width % 64
    height = height - height % 64
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent

#### ------------------Section Version 2 for Cross Attention Controll ------------------ 
## Helper function to get the Unet Cross Attention model
def init_attention_weights(tokenizer, unet, device, weight_tuples):
    tokens_length = tokenizer.model_max_length
    weights = torch.ones(tokens_length)
    
    for i, w in weight_tuples:
        if i < tokens_length and i >= 0:
            weights[i] = w
    
    
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.last_attn_slice_weights = weights.to(device)
        if module_name == "CrossAttention" and "attn1" in name:
            module.last_attn_slice_weights = None

def init_attention_func(unet):
    #ORIGINAL SOURCE CODE: https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L276
    def new_attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attn_slice = attention_scores.softmax(dim=-1)
        # compute attention output
        
        if self.use_last_attn_slice:
            if self.last_attn_slice_mask is not None:
                new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
            else:
                attn_slice = self.last_attn_slice

            self.use_last_attn_slice = False

        if self.save_last_attn_slice:
            self.last_attn_slice = attn_slice
            self.save_last_attn_slice = False

        if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
            attn_slice = attn_slice * self.last_attn_slice_weights
            self.use_last_attn_weights = False
        
        hidden_states = torch.matmul(attn_slice, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states
    
    def new_sliced_attention(self, query, key, value, sequence_length, dim):
        
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
            )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            
            if self.use_last_attn_slice:
                if self.last_attn_slice_mask is not None:
                    new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                    attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                else:
                    attn_slice = self.last_attn_slice
                
                self.use_last_attn_slice = False
                    
            if self.save_last_attn_slice:
                self.last_attn_slice = attn_slice
                self.save_last_attn_slice = False
                
            if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                attn_slice = attn_slice * self.last_attn_slice_weights
                self.use_last_attn_weights = False
            
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.last_attn_slice = None
            module.use_last_attn_slice = False
            module.use_last_attn_weights = False
            module.save_last_attn_slice = False
            module._sliced_attention = new_sliced_attention.__get__(module, type(module))
            module._attention = new_attention.__get__(module, type(module))

def use_last_tokens_attention_weights(unet, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_weights = use

@torch.no_grad()
def stablediffusion(model, prompt="",  prompt_edit_token_weights=[], init_latents=None, 
                        guidance_scale=7.5, steps=50, seed=None, width=512, height=512):
    #Change size to multiple of 64 to prevent size mismatches inside model
    if init_latents is not None:
        width = init_latents.shape[-1] * 8
        height = init_latents.shape[-2] * 8
        
    width = width - width % 64
    height = height - height % 64
    
    #If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None: seed = random.randrange(2**32 - 1)
    generator = torch.cuda.manual_seed(seed)
    #If init_latents is used, initialize noise as init_latent
    init_latent = torch.zeros((1, model.unet.in_channels, height // 8, width // 8), device=model.device)
    if init_latents is not None:
        noise = init_latents
    else:
        #Generate random normal noise
        noise = torch.randn(init_latent.shape, generator=generator, device=model.device)
    
    t_start = 0
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    init_latents = noise
    latent = scheduler.add_noise(init_latent, noise, torch.tensor([scheduler.timesteps[t_start]], device=model.device)).to(model.device)
        
        
    
    #Set inference timesteps to scheduler
    # scheduler= DDPMScheduler(beta_start= 0.00085, beta_end= 0.012, beta_schedule="scaled_linear", num_train_timesteps= 1000)
    # scheduler.set_timesteps(steps)
    # init_latent = torch.zeros((1, model.unet.in_channels, height // 8, width // 8), device=model.device)

    # bsz= init_latent.shape[0]
    # ## Sample a random timestep for each image 
    # timesteps= torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device= model.device).long() 
    # ## Add noise to the latents according to the noise magnitude at each timestep 
    # ## This is the forward diffusion process 
    # latent= scheduler.add_noise(init_latent, noise, timesteps)


    #Process clip
    with torch.autocast("cuda"):
        tokens_unconditional = model.tokenizer("", padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_unconditional = model.text_encoder(tokens_unconditional.input_ids.to(model.device)).last_hidden_state

        tokens_conditional = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_conditional = model.text_encoder(tokens_conditional.input_ids.to(model.device)).last_hidden_state


        init_attention_func(model.unet)
        init_attention_weights(model.tokenizer, model.unet, model.device, prompt_edit_token_weights)
            
        timesteps = scheduler.timesteps[t_start:]
        
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            t_index = t_start + i

            #sigma = scheduler.sigmas[t_index]
            latent_model_input = latent
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            #Predict the unconditional noise residual
            noise_pred_uncond = model.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
            
     
           
            #Use weights on non-edited prompt when edit is None
            use_last_tokens_attention_weights(model.unet)
                
            #Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = model.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
            
    
            #Perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latent = scheduler.step(noise_pred, t_index, latent).prev_sample

        #scale and decode the image latents with vae
        latent = latent / 0.18215
        image = model.vae.decode(latent.to(model.vae.dtype)).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)

# @torch.no_grad()
# def inversestablediffusion(model, init_image, prompt="", guidance_scale=3.0, steps=50, 
#                 refine_iterations=3,
#                 generator= torch.cuda.manual_seed(798122),
#                 device="cuda", 
#                 refine_strength=0.9, 
#                 refine_skip=0.7):
#     #Change size to multiple of 64 to prevent size mismatches inside model
#     width, height = init_image.size
#     width = width - width % 64
#     height = height - height % 64
    
#     image_width, image_height = init_image.size
#     left = (image_width - width)/2
#     top = (image_height - height)/2
#     right = left + width
#     bottom = top + height
    
#     init_image = init_image.crop((left, top, right, bottom))
#     init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
#     init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))

#     #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
#     if init_image.shape[1] > 3:
#         init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])

#     #Move image to GPU
#     init_image = init_image.to(model.device)

#     train_steps = 1000
#     step_ratio = train_steps // steps
#     timesteps = torch.from_numpy(np.linspace(0, train_steps - 1, steps + 1, dtype=float)).int().to(model.device)
    
#     betas = torch.linspace(0.00085**0.5, 0.012**0.5, train_steps, dtype=torch.float32) ** 2
#     alphas = torch.cumprod(1 - betas, dim=0)
    
#     init_step = 0
    
#     #Fixed seed such that the vae sampling is deterministic, shouldn't need to be changed by the user...
#     #generator = torch.cuda.manual_seed(798122)
    
#     #Process clip
#     with torch.autocast(device):
#         init_latent = model.vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215
        
#         tokens_unconditional = model.tokenizer("", padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
#         embedding_unconditional = model.text_encoder(tokens_unconditional.input_ids.to(model.device)).last_hidden_state

#         tokens_conditional = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
#         embedding_conditional = model.text_encoder(tokens_conditional.input_ids.to(model.device)).last_hidden_state
        
#         latent = init_latent

#         for i in tqdm(range(steps), total=steps):
#             t_index = i + init_step
            
#             t = timesteps[t_index]
#             t1 = timesteps[t_index + 1]
#             #Magic number for tless taken from Narnia, used for backwards CFG correction
#             tless = t - (t1 - t) * 0.25
            
#             ap = alphas[t] ** 0.5
#             bp = (1 - alphas[t]) ** 0.5
#             ap1 = alphas[t1] ** 0.5
#             bp1 = (1 - alphas[t1]) ** 0.5
            
#             latent_model_input = latent
#             #Predict the unconditional noise residual
#             noise_pred_uncond = model.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
            
#             #Predict the conditional noise residual and save the cross-attention layer activations
#             noise_pred_cond = model.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
            
#             #Perform guidance
#             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
#             #One reverse DDIM step
#             px0 = (latent_model_input - bp * noise_pred) / ap
#             latent = ap1 * px0 + bp1 * noise_pred
            
#             #Initialize loop variables
#             latent_refine = latent
#             latent_orig = latent_model_input
#             min_error = 1e10
#             lr = refine_strength
            
#             #Finite difference gradient descent method to correct for classifier free guidance, performs best when CFG is high
#             #Very slow and unoptimized, might be able to use Newton's method or some other multidimensional root finding method
#             if i > (steps * refine_skip):
#                 for k in range(refine_iterations):
#                     #Compute reverse diffusion process to get better prediction for noise at t+1
#                     #tless and t are used instead of the "numerically correct" t+1, produces way better results in practice, reason unknown...
#                     noise_pred_uncond = model.unet(latent_refine, tless, encoder_hidden_states=embedding_unconditional).sample
#                     noise_pred_cond = model.unet(latent_refine, t, encoder_hidden_states=embedding_conditional).sample
#                     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
#                     #One forward DDIM Step
#                     px0 = (latent_refine - bp1 * noise_pred) / ap1
#                     latent_refine_orig = ap * px0 + bp * noise_pred
                    
#                     #Save latent if error is smaller
#                     error = float((latent_orig - latent_refine_orig).abs_().sum())
#                     if error < min_error:
#                         latent = latent_refine
#                         min_error = error

#                     #print(k, error)
                    
#                     #Break to avoid "overfitting", too low error does not produce good results in practice, why?
#                     if min_error < 5:
#                         break
                    
#                     #"Learning rate" decay if error decrease is too small or negative (dampens oscillations)
#                     if (min_error - error) < 1:
#                         lr *= 0.9
                    
#                     #Finite difference gradient descent
#                     latent_refine = latent_refine + (latent_model_input - latent_refine_orig) * lr
                    
            
#     return latent

@torch.no_grad()
def inversestablediffusion(model, init_image, prompt="", guidance_scale=3.0, steps=50, refine_iterations=3, refine_strength=0.9, refine_skip=0.7):
    #Change size to multiple of 64 to prevent size mismatches inside model
    width, height = init_image.size
    width = width - width % 64
    height = height - height % 64
    
    image_width, image_height = init_image.size
    left = (image_width - width)/2
    top = (image_height - height)/2
    right = left + width
    bottom = top + height
    
    init_image = init_image.crop((left, top, right, bottom))
    init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
    init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))

    #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
    if init_image.shape[1] > 3:
        init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])

    #Move image to GPU
    init_image = init_image.to(model.device)

    train_steps = 1000
    step_ratio = train_steps // steps
    timesteps = torch.from_numpy(np.linspace(0, train_steps - 1, steps + 1, dtype=float)).int().to(model.device)
    
    betas = torch.linspace(0.00085**0.5, 0.012**0.5, train_steps, dtype=torch.float32) ** 2
    alphas = torch.cumprod(1 - betas, dim=0)
    
    init_step = 0
    
    #Fixed seed such that the vae sampling is deterministic, shouldn't need to be changed by the user...
    generator = torch.cuda.manual_seed(798122)
    
    #Process clip
    with torch.autocast("cuda"):
        init_latent = model.vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215
        
        tokens_unconditional = model.tokenizer("", padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_unconditional = model.text_encoder(tokens_unconditional.input_ids.to(model.device)).last_hidden_state

        tokens_conditional = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_conditional = model.text_encoder(tokens_conditional.input_ids.to(model.device)).last_hidden_state
        
        latent = init_latent

        for i in tqdm(range(steps), total=steps):
            t_index = i + init_step
            
            t = timesteps[t_index]
            t1 = timesteps[t_index + 1]
            #Magic number for tless taken from Narnia, used for backwards CFG correction
            tless = t - (t1 - t) * 0.25
            
            ap = alphas[t] ** 0.5
            bp = (1 - alphas[t]) ** 0.5
            ap1 = alphas[t1] ** 0.5
            bp1 = (1 - alphas[t1]) ** 0.5
            
            latent_model_input = latent
            #Predict the unconditional noise residual
            noise_pred_uncond = model.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
            
            #Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = model.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
            
            #Perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            #One reverse DDIM step
            px0 = (latent_model_input - bp * noise_pred) / ap
            latent = ap1 * px0 + bp1 * noise_pred
            
            #Initialize loop variables
            latent_refine = latent
            latent_orig = latent_model_input
            min_error = 1e10
            lr = refine_strength
            
            #Finite difference gradient descent method to correct for classifier free guidance, performs best when CFG is high
            #Very slow and unoptimized, might be able to use Newton's method or some other multidimensional root finding method
            if i > (steps * refine_skip):
                for k in range(refine_iterations):
                    #Compute reverse diffusion process to get better prediction for noise at t+1
                    #tless and t are used instead of the "numerically correct" t+1, produces way better results in practice, reason unknown...
                    noise_pred_uncond = model.unet(latent_refine, tless, encoder_hidden_states=embedding_unconditional).sample
                    noise_pred_cond = model.unet(latent_refine, t, encoder_hidden_states=embedding_conditional).sample
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    #One forward DDIM Step
                    px0 = (latent_refine - bp1 * noise_pred) / ap1
                    latent_refine_orig = ap * px0 + bp * noise_pred
                    
                    #Save latent if error is smaller
                    error = float((latent_orig - latent_refine_orig).abs_().sum())
                    if error < min_error:
                        latent = latent_refine
                        min_error = error

                    #print(k, error)
                    
                    #Break to avoid "overfitting", too low error does not produce good results in practice, why?
                    if min_error < 5:
                        break
                    
                    #"Learning rate" decay if error decrease is too small or negative (dampens oscillations)
                    if (min_error - error) < 1:
                        lr *= 0.9
                    
                    #Finite difference gradient descent
                    latent_refine = latent_refine + (latent_model_input - latent_refine_orig) * lr
                    
            
    return latent

# sampling_resize= {
#     "BILINEAR": PIL.Image.BILINEAR,
#     "BICUBIC": PIL.Image.BICUBIC,
#     "LANCZOS": PIL.Image.LANCZOS,
#     "NEAREST": PIL.Image.NEAREST,
#     "ANTIALIAS": PIL.Image.ANTIALIAS,
# }

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def check_inputs(prompt, strength): 
    if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [1.0, 1.0] but is {strength}")

def prepare_latents( model,image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        image = image.to(device=device, dtype=dtype)
        init_latent_dist = model.vae.encode(image).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt * num_images_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
        # get latents
        init_latents = model.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents
        return latents
def encode_prompt(model, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = model.tokenizer(prompt, padding="max_length", return_tensors="pt").input_ids

        if not torch.equal(text_input_ids, untruncated_ids):
            removed_text = model.tokenizer.batch_decode(untruncated_ids[:, model.tokenizer.model_max_length - 1 : -1])
         
        if hasattr(model.text_encoder.config, "use_attention_mask") and model.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = model.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = model.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(model.text_encoder.config, "use_attention_mask") and model.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = model.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

def init_image_to_latent(model, prompt: Union[str, List[str]], 
                            image: Union[torch.FloatTensor, PIL.Image.Image], 
                            negative_prompt: str= "",
                            strength=0.3,
                            generator=None, 
                            guidance_scale=7.5, 
                            num_inference_steps=50,  
                            num_images_per_prompt=1, 
                            ): 
        # 1. Check inputs
        check_inputs(prompt, strength)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = model.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = encode_prompt(model,
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Preprocess image
        if isinstance(image, PIL.Image.Image):
            image = preprocess(image)

        # 5. set timesteps
        model.scheduler.set_timesteps(num_inference_steps)
        timesteps, num_inference_steps = model.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = prepare_latents(
            model,image, latent_timestep, batch_size, num_images_per_prompt, text_embeddings.dtype, device, generator
        )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * model.scheduler.order
    
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = model.scheduler.step(noise_pred, t, latents,).prev_sample
           
        return latents 

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return self.to_out(out)

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count

def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)

def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int, word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha

def get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                                   tokenizer,):
    max_num_words=tokenizer.model_max_length
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words) # time, batch, heads, pixels, words
    
    
    return alpha_time_words
## Processing image to certain offset 
def process_image(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h > w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w > h:
        offset = (h - w) // 2
        image = image[offset:offset + w, :]
    ## Resize the image to 512x512
    image= np.array(Image.fromarray(image).resize((512, 512)))
    return image 




