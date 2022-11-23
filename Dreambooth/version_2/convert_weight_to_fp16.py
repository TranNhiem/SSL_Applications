from natsort import natsorted 
from glob import glob 
import argparse 
# from os.path import osp 
import os 
import torch 
from convert_diffusers_to_original_stable_diffusion import convert_unet_state_dict, convert_vae_state_dict, convert_text_enc_state_dict


parser = argparse.ArgumentParser()

parser.add_argument("--model_path", default="/data1/StableDiffusion/Dreambooth/rick/sd_v14", type=str, required=False, help="Path to the model to convert.")
parser.add_argument("--checkpoint_path", default="/data1/StableDiffusion/Dreambooth/rick/sd_v14_fp16.ckpt", type=str, required=False, help="Path to the output model.")
parser.add_argument("--half", default=True, action="store_true", help="Save weights in half precision.")


# weight_path= "/data1/StableDiffusion/Dreambooth/rick/sd_v14"
# WEIGHTS_DIR = natsorted(glob(weight_path + os.sep + "*"))[-1]
# print(WEIGHTS_DIR)

if __name__ == "__main__":
    args = parser.parse_args()
    assert args.model_path is not None, "Must provide a model path!"
    assert args.checkpoint_path is not None, "Must provide a checkpoint path!"

    unet_path = os.path.join(args.model_path, "unet", "diffusion_pytorch_model.bin")
    vae_path = os.path.join(args.model_path, "vae", "diffusion_pytorch_model.bin")
    text_enc_path = os.path.join(args.model_path, "text_encoder", "pytorch_model.bin")

    # Convert the UNet model
    unet_state_dict = torch.load(unet_path, map_location="cpu")
    unet_state_dict = convert_unet_state_dict(unet_state_dict)
    unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}

    # Convert the VAE model
    vae_state_dict = torch.load(vae_path, map_location="cpu")
    vae_state_dict = convert_vae_state_dict(vae_state_dict)
    vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

    # Convert the text encoder model
    text_enc_dict = torch.load(text_enc_path, map_location="cpu")
    text_enc_dict = convert_text_enc_state_dict(text_enc_dict)
    text_enc_dict = {"cond_stage_model.transformer." + k: v for k, v in text_enc_dict.items()}


    # Put together new checkpoint
    state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict}
    if args.half:
        state_dict = {k: v.half() for k, v in state_dict.items()}
    state_dict = {"state_dict": state_dict}
    torch.save(state_dict, args.checkpoint_path)