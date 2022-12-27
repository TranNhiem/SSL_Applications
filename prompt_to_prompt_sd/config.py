from absl_mock import Mock_Flag
import argparse
import os 
def read_cfg(): 
    flags= Mock_Flag() 
    base_cfg()

    return flags 

def base_cfg():

    flags = Mock_Flag() 

    flags.DEFINE_integer(
    'img_height', 512,
    'image height.')

    flags.DEFINE_integer(
    'img_width', 512,
    'image width.')

    flags.DEFINE_enum(
        "sd_model", "CompVis/stable-diffusion-v1-4", [ "CompVis/stable-diffusion-v1-4","prompthero/openjourney","runwayml/stable-diffusion-v1-5" ], 
        'The pretrained SD Model from Huggingface')
    
    flags.DEFINE_string(
        "store_path", "/data1/pretrained_weight/StableDiffusion/",
        'Path to store the download weight --> Saving in local disk')

    flags.DEFINE_integer(
    'num_diffusion_steps', 50,
    'The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.')

    flags.DEFINE_float(
    'guidance_scale', 7.5,
    'Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality')

    flags.DEFINE_boolean(
        'low_resource', False,  # 
        'Enable to True for running on 12GB GPUs ')

# Configure input Arguments for pretraining and output the pipeline 
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--ƒ",
        type=str,
        default="/data1/pretrained_weight/StableDiffusion/models--runwayml--stable-diffusion-v1-5/",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data1/pretrained_weight/StableDiffusion/",
        required=False,
        help="Path to save pretrained model to the Disk.",
    )
    
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        required=False,
        help="Path to input image to edit.",
    )
    parser.add_argument(
        "--target_text",
        type=str,
        default=None,
        help="The target text describing the output image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data1/StableDiffusion/Imagic/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--emb_train_steps",
        type=int,
        default=500,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--emb_learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizing the embeddings.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate for fine tuning the model.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
