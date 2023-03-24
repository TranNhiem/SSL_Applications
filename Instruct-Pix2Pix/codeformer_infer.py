import sys
sys.path.append('/home/rick/SSL_Application/SSL_Applications/inpainting_sd/CodeFormer')
import os
import cv2
import torch
import torch.nn.functional as F
import PIL
import numpy as np
from PIL import Image
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY

os.system("pip freeze")
pretrain_model_url = {
    'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
    'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
    'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth',
    "realesrgan4x": 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
}
# download weights
if not os.path.exists('CodeFormer/weights/CodeFormer/codeformer.pth'):
    load_file_from_url(url=pretrain_model_url['codeformer'], model_dir="/media/rick/2TB_2/pretrained_weight/Codeformer/CodeFormer/", progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/facelib/detection_Resnet50_Final.pth'):
    load_file_from_url(url=pretrain_model_url['detection'], model_dir='/media/rick/2TB_2/pretrained_weight/Codeformer/facelib/', progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/facelib/parsing_parsenet.pth'):
    load_file_from_url(url=pretrain_model_url['parsing'], model_dir='/media/rick/2TB_2/Codeformer/facelib/', progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth'):
    load_file_from_url(url=pretrain_model_url['realesrgan'], model_dir="/media/rick/2TB_2/pretrained_weight/Codeformer/realesrgan/", progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/realesrgan/RealESRNet_x4plus.pth'):
    load_file_from_url(url=pretrain_model_url['realesrgan4x'], model_dir="/media/rick/2TB_2/pretrained_weight/Codeformer/realesrgan/", progress=True, file_name=None)

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# set enhancer with RealESRGAN
def set_realesrgan(model_type="2x"):
    half = True if torch.cuda.is_available() else False

    if model_type =="4x":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
        scale=4,
        model_path="/media/rick/2TB_2/pretrained_weight/Codeformer/realesrgan/RealESRNet_x4plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    else: 
        model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
        upsampler = RealESRGANer(
            scale=2,
            model_path="/media/rick/2TB_2/pretrained_weight/Codeformer/realesrgan/RealESRGAN_x2plus.pth",
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=half,
        )
    return upsampler



device= torch.device("cuda" if torch.cuda.is_available() else 'cpu')
def set_codeformer():
    codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, 
        codebook_size= 1024, 
        n_head= 8, 
        n_layers= 9, 
        connect_list=["32", "64", "128", "256"]
    ).to(device)
    ckpt_path = "/media/rick/2TB_2/pretrained_weight/Codeformer/CodeFormer/codeformer.pth"
    checkpoint = torch.load(ckpt_path)["params_ema"]
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()
    return codeformer_net



os.makedirs('/home/rick/BLIR/SSL_Applications/inpainting_sd/output', exist_ok=True)

def inference(image, background_enhance, face_upsample, upscale, codeformer_fidelity, model_type="2x"): 
    try: 
        has_aligned= False
        only_center_face= False
        draw_box= False 
        detection_model= "retinaface_resnet50"

        if isinstance(image, PIL.Image.Image):
            img = np.array(image)
            print('\timage size:', img.shape)
        elif isinstance(image, str):
            print('Inp:', image, background_enhance, face_upsample, upscale, codeformer_fidelity)
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            print('\timage size:', img.shape)
        else: 
            raise ValueError("Image should be PIL.Image or str")

        upscale= int(upscale)
        
        if upscale > 6:
            upscale = 6  # avoid momory exceeded due to too large upscale
        if upscale > 2 and min(img.shape[:2])>1280:
            upscale = 2  # avoid momory exceeded due to too large img resolution

        face_helper = FaceRestoreHelper(
            upscale, 
            face_size=512, 
            crop_ratio= (1, 1), 
            det_model= detection_model,
            save_ext="png", 
            use_parse=True, 
            device= device, 
        )
        upsampler = set_realesrgan(model_type=model_type)
        bg_upsampler= upsampler if background_enhance else None
        face_upsampler= upsampler if face_upsample else None

        if has_aligned: 
            # the input faces are already cropped and aligned
            img= cv2.resize(img, (512, 512), interpolation= cv2.INTER_LINEAR)
            face_helper.is_gray= is_gray(img, threshold=5)
            if face_helper.is_gray:
                print('\tgrayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            print(f'\tdetect{num_det_faces} faces')
            ## Align and warp each face 
            face_helper.align_warp_face()

        ## Face restoration for each cropping face 
        for idx, cropped_face in enumerate(face_helper.cropped_faces): 
            ## Prepare data
            cropped_face_t= img2tensor(cropped_face/255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t= cropped_face_t.unsqueeze(0).to(device)
            try: 
                codeformer_net=set_codeformer()
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                    del codeformer_net
                 
                    torch.cuda.empty_cache()

            except  RuntimeError as e:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )
            retored_face= restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        ## Paste back to reconstruct the whole image
        if not has_aligned: 
            # upsample the background 
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None

            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img= face_helper.paste_faces_to_input_image(upsample_img= bg_img,
                draw_box= draw_box, 
                face_upsampler= face_upsampler,
                )
            else: 
                restored_img= face_helper.paste_faces_to_input_image(upsample_img= bg_img,draw_box= draw_box)
            del face_upsampler
            del bg_upsampler
            torch.cuda.empty_cache()
        # save restored img
        # save_path = f'/home/harry/BLIRL/SSL_Applications/inpainting_sd/upscale_img0.png'
        # imwrite(restored_img, str(save_path))
        restored_img = cv2.cvtColor(restored_img,  cv2.COLOR_BGR2RGB)
        restored_img=Image.fromarray(restored_img)

        return restored_img 
        
    except Exception as error:
        print('Global exception', error)
        return None, None