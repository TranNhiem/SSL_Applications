import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch

import torchvision
import pytorch_lightning as 
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

## Create the some initialization 
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n","--name",type=str,const=True,default="",nargs="?",help="postfix for logdir",)
    parser.add_argument( "-r","--resume",type=str,const=True, default="",nargs="?",help="resume from logdir or checkpoint in logdir",)
    parser.add_argument( "-b","--base",nargs="*",metavar="base_config.yaml",help="paths to base configs. Loaded from left-to-right. " "Parameters can be overwritten or added with command-line options of the form `--key value`.",default=list(),)
    parser.add_argument( "-t", "--train",type=str2bool, const=True, default=False, nargs="?", help="train", )
    parser.add_argument( "--no-test",type=str2bool, const=True, default=False, nargs="?", help="disable test", )
    parser.add_argument(  "-p","--project",help="name of new or path to existing project")
    parser.add_argument( "-d", "--debug",  type=str2bool,  nargs="?", const=True, default=False, help="enable post-mortem debugging", )
    parser.add_argument( "-s", "--seed", type=int, default=23, help="seed for seed_everything", )
    parser.add_argument( "-f", "--postfix",type=str,default="", help="post-postfix for default name",)
    parser.add_argument( "-l", "--logdir",type=str,default="logs",help="directory for logging dat shit",)
    parser.add_argument("--scale_lr", type=str2bool,nargs="?", const=False, default=False, help="scale base-lr by ngpu * batch_size * n_accumulate",)
    parser.add_argument("--datadir_in_name",type=str2bool,  nargs="?", const=True, default=True, help="Prepend the final directory in the data_root to the output directory name")

    parser.add_argument("--actual_resume", type=str,required=True, help="Path to model to actually resume from")
    parser.add_argument("--data_root",  type=str,  required=True, help="Path to directory with training images")
    
    parser.add_argument("--reg_data_root", 
        type=str, 
        required=True, 
        help="Path to directory with regularization images")

    parser.add_argument("--embedding_manager_ckpt", 
        type=str, 
        default="", 
        help="Initialize embedding manager from a checkpoint")

    parser.add_argument("--class_word", 
        type=str, 
        default="dog",
        help="Placeholder token which will be used to denote the concept in future prompts")

    parser.add_argument("--init_word", 
        type=str, 
        help="Word to use as source for initial token embedding")

    return parser


## Checking loading model with Given Checkpoint
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    return model