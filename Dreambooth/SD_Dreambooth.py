## TranNhiem 2022-11-13 
'''
Building Dreambooth model from scratch -- Finetuning Diffusion model 
    1. Unet model 
    2. VAE 
    3. Diffusion model

## Requirement 
#@title Install the required libs
!pip install -qq diffusers==0.6.0 accelerate tensorboard transformers ftfy gradio
!pip install -qq "ipywidgets>=7,<8"
!pip install -qq bitsandbytes

'''

import gc 
import math
from loading_concept_image import PromptDataset, DreamBoothDataset 
from pathlib import Path
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel 
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import torch 
import torch.nn.functional as F
from argparse import Namespace

import bitsandbytes as bnb
##----------------- Temporal Setting Hyperparameters -----------------##


# def created_concpet(instance_prompt= "A photo of sks rick", prior_presevation= False, prior_preservation_class_prompt= None,
#                      prior_loss_weight=0.5, num_class_images=1): 
#     '''
#     ## instance prompt is a prompt that describes of what your object or style is, together with the initializer work "sks"

#     '''

### instance prompt is a prompt that describes of what your object or style is, together with the initializer work "sks"
instance_prompt = "a photo of sks rick" #@param {type:"string"}
#@markdown Check the `prior_preservation` option if you would like class of the concept (e.g.: toy, dog, painting) is guaranteed to be preserved. This increases the quality and helps with generalization at the cost of training time

prior_preservation_class_prompt = "a photo of a cat clay toy" #@param {type:"string"}
sample_batch_size = 2
prior_loss_weight = 0.5 ## weight for each concept class ex: cat_concept 0.1, dog_concept 0.1, toy_concept 0.8
prior_preservation_class_folder = "/data1/StableDiffusion/Dreambooth/cat_clay_toy"
class_data_root=prior_preservation_class_folder
class_prompt=prior_preservation_class_prompt
device="cuda"


##----------------- Generate Class Images if Not Provided -----------------##

# "CompVis/stable-diffusion-v1-4",
# "runwayml/stable-diffusion-v1-5"
# "runwayml/stable-diffusion-inpainting",
ouput_dir = Path("/data1/StableDiffusion/Dreambooth/").mkdir(parents=True, exist_ok=True)
data_path="/home/harry/BLIRL/SSL_Applications/Dreambooth/Rick/"
pretrain_model_name_or_path="runwayml/stable-diffusion-v1-5"
num_class_images = 12 
prior_preservation = False #@param {type:"boolean"}
if (prior_preservation):
    #print("Generating class images...")
    class_images_dir = Path(class_data_root)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    if cur_class_images < num_class_images: 
        pipeline= StableDiffusionPipeline.from_pretrained(pretrain_model_name_or_path,
                                                        revision="fp16", torch_dtype=torch.float16).to(device)
        # Using this for save memory
        ##pipeline.enable_attention_slicing()
        pipeline.set_progress_bar_config(disable=True)
        num_new_images= num_class_images - cur_class_images
        print(f"Number of class images to sample: {num_new_images}")

        sample_dataset= PromptDataset(class_prompt, num_new_images)
        sample_dataloader= torch.utils.data.DataLoader(sample_dataset, batch_size=sample_batch_size, shuffle=False, num_workers=10)
        
        for example in tqdm(sample_dataloader, desc="Generating class images"): 
            images= pipeline(example["prompt"]).images 
            for i, image in enumerate(images):
                image.save(class_images_dir / f"{example['index'][i]+ cur_class_images}.jpg")
        pipeline=None 
        gc.collect() 
        del pipeline 
        with torch.no_grad():
            torch.cuda.empty_cache()

##----------------- Load multiple Pretrained Models -----------------##
## Loading seperate each Encoder 
text_encoder = CLIPTextModel.from_pretrained(pretrain_model_name_or_path, subfolder="text_encoder")
vae= AutoencoderKL.from_pretrained(pretrain_model_name_or_path, subfolder="vae")
unet= UNet2DConditionModel.from_pretrained(pretrain_model_name_or_path, subfolder="unet")
tokenizer= CLIPTokenizer.from_pretrained(pretrain_model_name_or_path, subfolder="tokenizer")

## ----------------- Training Pipeline -----------------##
args=Namespace(
    pretrained_model_name_or_path=pretrain_model_name_or_path,
    resolution=512, 
    center_crop=True, 
    instance_data_dir=data_path , 
    instance_prompt= instance_prompt, 
    learning_rate= 5e-6, 
    max_train_steps= 450, 
    train_batch_size= 1, 
    gradient_accumulation_steps= 1,
    max_grad_norm= 1.0,
    mixed_precision= "fp16",#fp16, fp32 for training. 
    gradient_checkpointing= True,# set to True to lower the memory usage 
    use_8bit_adam=True, #use abit optimizer from bisandbytes 
    seed= 3434554, 
    class_data_dir= prior_preservation_class_folder, 
    class_prompt= prior_preservation_class_prompt,
    num_class_images= num_class_images,
    output_dir=ouput_dir ,
    with_prior_preservation= prior_preservation,
    prior_preservation_class_folder= prior_preservation_class_folder,
    noise_schedule= "ddpm", #ddpm, pndms
)

def training_function(text_encoderm=text_encoder, vae=vae, unet=unet, tokenizer=tokenizer, args=args):
    logger= get_logger(__name__)
    accelerator= Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, 
                                mixed_precision=args.mixed_precision)
    set_seed(args.seed)
    if args.gradient_checkpointing:
        logger.info("Using gradient checkpointing")
        unet.enable_gradient_checkpointing() ## New update from diffusers model of the Unet model 
    ## Use 8-Bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs 
    if args.use_8bit_adam:
        optimizer_cls= bnb.optim.AdamW8bit
    else: 
        optimizer_cls= torch.optim.AdamW 
    
    optimizer= optimizer_cls(
                            unet.parameters(), # Only Update the Unet Model
                            lr=args.learning_rate)

    ##-----------------Noise Schedule Function -----------------##
    if args.noise_schedule == "ddpm":
        noise_scheduler= DDPMScheduler(beta_start= 0.00085, beta_end= 0.012, beta_schedule="scaled_linear", num_train_timesteps= 1000)
    elif args.noise_schedule == "pndms":
        noise_scheduler= PNDMScheduler(beta_start= 0.00085, beta_end= 0.012, beta_schedule="scaled_linear", num_train_timesteps= 1000)
    else: 
        raise ValueError("Noise schedule not supported")

    ##----------------- Loading and Processing Data -----------------##
    train_dataset= DreamBoothDataset(root_dir=args.instance_data_dir, 
                                    instance_prompt= args.instance_prompt, # name of your concept
                                    class_data_root= args.class_data_dir if args.with_prior_preservation else None,
                                    tokenizer= tokenizer,
                                    class_prompt= args.class_prompt,
                                    size= args.resolution,
                                    transform=None, 
                                    center_crop= args.center_crop, 
                                    )
    ## Helper function to continue processing image and text data.                              
    def collate_fn(examples): 

        input_ids= [example["instance_prompt_ids"] for example in examples]
        pixel_values= [example["instance_images"] for example in examples]

        ##  Concatenate the class and instance examples for prior preservation 
        if args.with_prior_preservation:
            ## If you have different class concept want to Fine-tune at the same time 
            ## Usage example Family with 5 members then --> 5 different classes 
            input_ids=[example["class_prompt_ids"] for example in examples]
            pixel_values= [example["instance_images"] for example in examples]

        pixel_values= torch.stack(pixel_values,)## stack --> [num_images, [3, 512, 512]]
        pixel_values= pixel_values.to(memory_format= torch.contiguous_format).float()
        input_ids= tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        batch= {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch 

    train_dataloader= torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, 
                                            shuffle=True, collate_fn=collate_fn, num_workers=10)
    #breakpoint()
    ## prepare for training Unet model 
    unet, optimizer,train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)#train_dataloader, train_dataloader
    ## move text_encoder and vae models to GPU 
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    ## need to recalculate of total training steps as the size of the training dataloader is changed 
    num_update_steps_per_epoch= math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs= math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    ## Training 
    total_batch_size= args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
        
    ## Using Hugging face accelerator to train the model --> Future Version using Pytorch Lightning 
    ## Only show the progress bar once on each machine... 
    progress_bar= tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("steps")
    global_step= 0 

    for epoch in range(num_train_epochs): 
        unet.train() 
        
        for step, batch in enumerate(train_dataloader): 
            #breakpoint()
            with accelerator.accumulate(unet): 
                ## Convert images to latent space 
                with torch.no_grad(): 
                    latents= vae.encode(batch["pixel_values"]).latent_dist.sample() 
                    latents= latents * 0.18215 
                ## Sample noise that we will add to the latents 
                noise= torch.randn(latents.shape).to(latents.device)
                bsz= latents.shape[0]
                ## Sample a random timestep for each image 
                timesteps= torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device= latents.device).long() 
                ## Add noise to the latents according to the noise magnitude at each timestep 
                ## This is the forward diffusion process 
                noisy_latents= noise_scheduler.add_noise(latents, noise, timesteps)

                ## get the text embedding for condition 
                with torch.no_grad(): 
                    encoder_hidden_states= text_encoder(batch["input_ids"])[0]
                ## Predict the noise residual 
                noise_pred= unet(noisy_latents,timesteps, encoder_hidden_states).sample 

                if args.with_prior_preservation: 
                    ## Chunk the noise and noise_pred into 2 parts and compute the loss on each part seperately. 
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0) 
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    ## Compute the instance loss 
                    loss = F.mse_loss(noise_pred_prior, noise_prior, reduction="none").mean([1, 2,3]).mean() 
                    ## Compute the piror loss 
                    prior_loss = F.mse_loss(noise_pred_prior, noise_prior, reduction="none").mean([1, 2,3]).mean()

                    ## Add the prior loss the the instance loss 
                    loss += prior_loss * args.prior_loss_weight
                else: 
                    loss =F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2,3]).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients: 
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step() 
                optimizer.zero_grad()
            
            ## check if the accelerator has performed an optimization step behine the scenes 
            if accelerator.sync_gradients: 
                progress_bar.update(1)
                global_step +=1 
            
            logs= {"loss": loss.detach().item()}
            progress_bar.set_postfix(logs)

            if global_step >= args.max_train_steps: 
                break
        accelerator.wait_for_everyone()

    #$ Save the model
    if accelerator.is_main_process:
        logger.info("Saving model checkpoint to %s", args.output_dir)
        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,

            scheduler=PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
            ) if args.scheduler == "pndm" else None,
            safety_checker=None,#StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )
        pipeline.save_pretrained(args.output_dir)



if __name__ == '__main__':

    training_function()
    