
import os
import sys
sys.path.append('./monodepth2')


from PIL import Image
import numpy as np
import cv2
from layers import disp_to_depth
from evaluate_depth import STEREO_SCALE_FACTOR



class Settings():
    def __init__(self) -> None:
        self.prediction_path = 'fileoutput/NPY_Depth'
        self.ground_truth_path = 'data_ass3/Task1_3/groundtruth'
        self.min_depth = 0.001
        self.max_depth = 80

def import_depth_info(path):
    npy_files = [f for f in os.listdir(path) if f.endswith('.npy')]
    arrays = [np.squeeze(np.load(os.path.join(path, f)), axis=0) for f in npy_files]
    concatenated_array = np.concatenate(arrays, axis=0)
    return concatenated_array

def import_gt_info(path):

    png_files = [f for f in os.listdir(path) if f.endswith('.png')]
    arrays = []
    
    for f in png_files:
        img = cv2.imread(os.path.join(path, f), cv2.IMREAD_UNCHANGED)
        img_array = np.array(img)
        arrays.append(img_array)

    concatenated_array = np.stack(arrays, axis=0)
    
    return concatenated_array

def error_eval(opt):

    pred_depths = import_depth_info(opt.prediction_path)
    gt_depths = import_gt_info(opt.ground_truth_path)


    print(f' Predicted value shape: {pred_depths.shape}')
    print(f' Groundtruth value shape: {gt_depths.shape}')

    errors = []
    for i in range(pred_depths.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        mask = gt_depth > 0


        pred = pred_depths[i]
        pred = cv2.resize(pred, (gt_width, gt_height), interpolation= 2)

        gt_depth = (gt_depth.astype(np.float32) / 256.0)
        _ , gt_depth = disp_to_depth(gt_depth, opt.min_depth, opt.max_depth)
        gt_depth *= STEREO_SCALE_FACTOR


        pred = pred[mask]
        gt_depth = gt_depth[mask]

        rmse = (gt_depth - pred) ** 2
        rmse = np.sqrt(rmse.mean())
        errors.append(rmse)

    mean_rmse = np.array(errors).mean()
    print(f' RMSE: {mean_rmse}')

if __name__ == "__main__":
    settings = Settings()
    error_eval(settings)