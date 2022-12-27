import PIL
from PIL import Image
import torch 
import cv2 
from codeformer_infer import inference
import numpy as np

#img= Image.open('/home/harry/BLIRL/SSL_Applications/inpainting_sd/img0.png')
img="/home/harry/BLIRL/SSL_Applications/inpainting_sd/img0.png"
upscale_img=inference(img, background_enhance= True, face_upsample= True, upscale= 6, codeformer_fidelity= 1.0, model_type="4x")
# imag=np.array(upscale_img)
# upscale_img_pil=Image.fromarray(imag)
upscale_img.save("/home/harry/BLIRL/SSL_Applications/inpainting_sd/upscale_img0.png")
#cv2.imwrite("/home/harry/BLIRL/SSL_Applications/inpainting_sd/upscale_img0.png", upscale_img)
del upscale_img
torch.cuda.empty_cache()