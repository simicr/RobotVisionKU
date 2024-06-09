
import os
import re
import sys
import numpy as np
import cv2


class Settings2():
    def __init__(self, prediction_path='fileoutput/NPY_Depth', 
                 ground_truth_path='data_ass3/Task1_3/groundtruth', ext='png') -> None:
        
        self.prediction_path = prediction_path
        self.ground_truth_path = ground_truth_path
        self.ext = ext

        self.min_depth = 0.001
        self.max_depth = 80


def import_depth_info(path, ext):
    
    npy_files = [f for f in sorted(os.listdir(path)) if f.endswith(ext)]

    arrays = [np.squeeze(np.load(os.path.join(path, f)), axis=0) for f in npy_files]
    
    concatenated_array = np.concatenate(arrays, axis=0)
    return concatenated_array, npy_files

def import_gt_info(path):

    png_files = [f for f in sorted(os.listdir(path)) if f.endswith('.png')]
    arrays = []
    
    for f in png_files:
        img = cv2.imread(os.path.join(path, f), cv2.IMREAD_UNCHANGED)
        img_array = np.array(img)
        arrays.append(img_array)

    concatenated_array = np.stack(arrays, axis=0)
    
    return concatenated_array, png_files

def error_eval(opt):

    pred_depths, files = import_depth_info(opt.prediction_path, opt.ext)
    gt_depths, _ = import_gt_info(opt.ground_truth_path)

    errors = []
    for i in range(pred_depths.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        mask = gt_depth > 0

        pred = pred_depths[i]
        pred = cv2.resize(pred, (gt_width, gt_height))[mask]
        pred[pred < settings.min_depth] = settings.min_depth
        pred[pred > settings.max_depth] = settings.max_depth

        gt_depth = ((gt_depth.astype(np.float32) / 256.0)[mask])

        rmse = np.sqrt(((gt_depth - pred) ** 2).mean())
        errors.append(rmse)

        print(f'{i + 1}. {" ".join(files[i].split("_")[:-1])} : {rmse}')

    mean_rmse = np.array(errors).mean()
    print(f'Avg. RMSE: {mean_rmse}')

    return errors

if __name__ == "__main__":
    settings = Settings2()
    error_eval(settings)