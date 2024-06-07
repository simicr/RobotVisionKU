from __future__ import absolute_import, division, print_function

import os
import sys
sys.path.append('./monodepth2')

import glob
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
import networks as networks

from torchvision import transforms
from layers import disp_to_depth
from evaluate_depth import STEREO_SCALE_FACTOR


class Settings():
    def __init__(self) -> None:
        self.image_path = 'data_ass3/Task1_3/images'
        self.model_name = 'mono+stereo_640x192'
        self.output_path = 'fileoutput'
        self.ext = 'png'
        self.min_depth = 0.001
        self.max_depth = 80
        self.no_cuda = True
        self.pred_metric_depth = True

def predict_depth(settings):

    model_path = os.path.join("models", settings.model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    paths = glob.glob(os.path.join(settings.image_path, '*.{}'.format(settings.ext)))
    output_directory = settings.output_path

    if torch.cuda.is_available() and not settings.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", model_path)
    

    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, True)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder( num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()
    

    with torch.no_grad():
        for _, image_path in enumerate(paths):

            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate( disp, (original_height, original_width), 
                                                           mode="bilinear", align_corners=False)

            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, settings.min_depth, settings.max_depth)
            
            if settings.pred_metric_depth:
                name_depth_npy = os.path.join(output_directory, "NPY_Depth" ,"{}_depth.npy".format(output_name))
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                np.save(name_depth_npy, metric_depth)
            
            name_dest_npy = os.path.join(output_directory, "NPY_Disp" , "{}_disp.npy".format(output_name))
            np.save(name_dest_npy, disp.cpu().numpy())

            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "IMG" ,"{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

    print('-> Done!')

if __name__ == '__main__':
    settings = Settings()
    predict_depth(settings)