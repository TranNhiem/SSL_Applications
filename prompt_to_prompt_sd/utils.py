import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm 

def text_under_image(image: np.ndarray, text:str, text_color: Tuple[int, int, int]=(0, 0, 0)):
    h, w, c= image.shape
    offset= int(h* .2)
    img= np.ones((h+ offset, w, c), dtype= np.unit8)* 255 
    font= cv2.FONT_HERSHEY_SIMPLEX
    img[:h]= image
    textsize= cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y= (w-textsize[0])//2, h+ (offset-textsize[1])//2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return np.array(image)

def view_images(images, num_rows=1 , offset_ratio=0.02): 
    if type(images) is list: 
        num_empty= len(images)% num_rows 
    elif images.ndim == 4: 
        num_empty= images.shape[0]% num_rows
    else: 
        images= [images]
        num_empty= 0
    empty_images= np.ones(images[0].shape, dtype=np.uint8)* 255 
    images= [image.astype(np.uint8) for image in images] + [empty_images]* num_empty
    num_items= len(images)

    h, w, c= images[0].shape 
    offset= int(h* offset_ratio)
    num_cols= num_items //num_rows 
    image_ = np.ones((h* num_rows+ offset* (num_rows-1), w* num_cols+ offset* (num_cols-1), c), dtype=np.uint8)* 255

    for i in range(num_rows):
        for j in range(num_cols):
            image_[i*(h+offset):i*(h+offset)+h, j*(w+offset):j*(w+offset)+w]= images[i*num_cols+j]
    pil_img= Image.fromarray(image_)
    display(pil_img)

# Create diffusion step function with arguments input model, controller, latents, t, guidance_scale, low_resource 
def make_diffusion_step(model, controller, latents, t, guidance_scale, low_resource=False):
    if low_resource: 
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]

    else: 
        latents_input= torch.cat([latents]* 2)
        noise_pred= model.unet (latents_input, t, encoder_hidden_states= context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred= noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents 

# Create latent to image function with arguments input vae model and latents  as input 
def latent_to_image(vae, latents):
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
    return latent, 

@torch.no_grad()
def text2image_sd(model, prompt: List[str], controller, num_inference_steps: int= 50 
                guidance_scale: float = 7.5, generator: Optional[torch.Generator]= None, 
                latent: Optional[torch.FloatTensor]= None, 
                low_resource: bool = False,  height =512, width= 512 
                
                ): 
    
    register_attention_control(model, controller)
    batch_size= len(prompt)
    text_input= model.tokenizer(prompt, padding= "max_length", 
                truncation=True, 
                return_tensors= "pt"
                )
    text_embeddings= model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input= model.tokenizer([""]* batch_size, padding="max_length", return_tensors="pt")
    uncond_embeddings= model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context= [uncond_embeddings, text_embeddings]
    if not low_resource: 
        context= torch.cat(context)
    latent, latents= init_latent(latent, model, height, width, generator, batch_size)

    ## Set timesteps 
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    for t in tqdm(model.scheduler.timesteps): 
        latents= make_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)

    image = latent_to_image(model.vae, latents)

    return image, latent

def register_attention_control(model, controller): 
    def ca_forward(self,place_in_unet): 
        def forward(x, context=None, mask=None): 
            batch_size, sequence_length, dim= x.shape 
            h = self.heads
            q= self.to_q(x)
            is_cross = context is not None 
            context = context if is_cross else x 
            k= self.to_k(context)
            v= self.to_v(context)
            q= self.reshape_heads_to_batch_dim(q)
            k= self.reshape_heads_to_batch_dim(k)
            v= self.reshape_heads_to_batch_dim(v)

            sim= torch.einsum("b i d, b j d -> b i j", q, k)* self.scale

            if mask is not None: 
                mask= mask.reshape(batch_size, -1)
                max_neg_value= -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)
            
            ## attention, what we can't get enought 
            attn = sim.softmax(dim=-1)
            attn= controller(attn, is_cross, place_in_unet)
            out= torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return self.to_out(out)
        return forward
    
    def register_recr(net_, count, place_in_unet): 
        if net_.__class__.__name__ == 'CrossAttention': 
            net_.forward= ca_forward(net_, place_in_unet)

            return count +1 
        elif hasattr(net_, 'children'): 
            for net__ in net_.children(): 
                count= register_recr(net__, count, place_in_unet)
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
                                   tokenizer, max_num_words=77):
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


def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE)
    ptp_utils.view_images(images)
    return images, x_t