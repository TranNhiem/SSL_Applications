export CUDA_VISIBLE_DEVICES=3,4
accelerate launch --config_file /home/harry/BLIRL/SSL_Applications/Dreambooth/version_1/accelerator_train_config.yaml --main_process_port 1234 SD_Dreambooth.py \
--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"\
--output_dir "/data1/StableDiffusion/Dreambooth/pretrained/rick_v2" \
