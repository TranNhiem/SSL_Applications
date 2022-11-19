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
import hashlib
import gc 
import math
from loading_concept_image import PromptDataset, DreamBoothDataset 
from pathlib import Path
import os 
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel 
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import torch 
import torch.nn.functional as F
from argparse import Namespace
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import bitsandbytes as bnb


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


##----------------- Temporal Setting Hyperparameters -----------------##


### instance prompt is a prompt that describes of what your object or style is, together with the initializer work "sks"
#@param {type:"string"}
#@markdown Check the `prior_preservation` option if you would like class of the concept (e.g.: toy, dog, painting) is guaranteed to be preserved. This increases the quality and helps with generalization at the cost of training time


## weight for each concept class ex: cat_concept 0.1, dog_concept 0.1, toy_concept 0.8
#prior_preservation_class_folder = "/home/harry/BLIRL/SSL_Applications/Dreambooth/regularize_img/rick_1"
# class_data_root=prior_preservation_class_folder

##----------------- Generate Class Images if Not Provided -----------------##

# "CompVis/stable-diffusion-v1-4",
# "runwayml/stable-diffusion-v1-5"
# "runwayml/stable-diffusion-inpainting",
#output_dir = Path("/data1/StableDiffusion/Dreambooth/pretrained/").mkdir(parents=True, exist_ok=True)

# pretrain_model_name_or_path="runwayml/stable-diffusion-v1-5"
#pretrain_model_name_or_path="runwayml/stable-diffusion-v1-5"



## ----------------- Training Hyperparameter Setting -----------------##
args=Namespace(
    pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse", 
    data_path="/home/harry/BLIRL/SSL_Applications/Dreambooth/Rick/",
    instance_prompt = "a photo of sks rick" , 
    resolution=512, 
    center_crop=True, 
    learning_rate= 5e-6, 
    max_train_steps= 6000, 
    train_batch_size= 4, 
    sample_batch_size=1,
    gradient_accumulation_steps= 1,
    max_grad_norm= 1.0,
    revision="fp16",
    mixed_precision= "fp16",#fp16, fp32 for training. 
    gradient_checkpointing= False,# set to True to lower the memory usage 
    use_8bit_adam=False, #use abit optimizer from bisandbytes 
    seed= 3434554, 
    class_data_dir= "/home/harry/BLIRL/SSL_Applications/Dreambooth/regularize_img/rick_1", 
    class_prompt=  "a photo of a rick",
    num_class_images= 100,
    prior_loss_weight = 0.5, # weight for prior loss 
    output_dir= "/data1/StableDiffusion/Dreambooth/pretrained/rick_v2", 
    with_prior_preservation= True,
    noise_schedule= "ddpm", #ddpm, pndms
)

def training_function(args=args):
    logger= get_logger(__name__)
    ##----------------- Load multiple Pretrained Models -----------------##
    accelerator= Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, 
                                mixed_precision=args.mixed_precision, )
    prior_preservation = True #@param {type:"boolean"}
    if (prior_preservation):
        #print("Generating class images...")
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images: 

            pipeline = None
        
            class_images_dir = Path(args.class_data_dir)
            class_images_dir.mkdir(parents=True, exist_ok=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < args.num_class_images:
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                if pipeline is None:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=AutoencoderKL.from_pretrained(
                            args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
                            subfolder=None if args.pretrained_vae_name_or_path else "vae",
                            revision=None if args.pretrained_vae_name_or_path else args.revision,
                            torch_dtype=torch_dtype
                        ),
                        torch_dtype=torch_dtype,
                    safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker", cache_dir="/data1/StableDiffusion/Dreambooth/"),
                        revision=args.revision
                    )
                    pipeline.set_progress_bar_config(disable=True)
                    pipeline.to(accelerator.device)

                num_new_images = args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(args.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.train_batch_size)

                sample_dataloader = accelerator.prepare(sample_dataloader)

                with torch.autocast("cuda"), torch.inference_mode():
                    for example in tqdm(
                        sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                    ):
                        images = pipeline(example["prompt"]).images

                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            image.save(image_filename)


    ##----------------- Load multiple Pretrained Models -----------------##
    ## Loading seperate each Encoder 
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae= AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet= UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    tokenizer= CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")



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
        print("You need to adjust Hyperparameters")
        noise_scheduler= PNDMScheduler(beta_start= 0.00085, beta_end= 0.012, beta_schedule="scaled_linear", num_train_timesteps= 1000)
    else: 
        raise ValueError("Noise schedule not supported")

    ##----------------- Loading and Processing Data -----------------##
    train_dataset= DreamBoothDataset(root_dir=args.data_path, 
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
            input_ids +=[example["class_prompt_ids"] for example in examples]
            pixel_values += [example["instance_images"] for example in examples]

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
            ) if args.noise_schedule == "ddpm" else None,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )

        pipeline.save_pretrained(args.output_dir)


if __name__ == '__main__':

    training_function()
    