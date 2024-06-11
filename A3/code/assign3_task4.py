import os
from assign3_task1 import Settings1, predict_depth
from assign3_task2 import Settings2, error_eval, import_depth_info
import numpy as np
import cv2
from pathlib import Path
import struct

import matplotlib.pyplot as plt 


def import_gt_info(path):

    png_files = [f for f in sorted(os.listdir(path)) if f.endswith('.png')]
    arrays = []
    
    for f in png_files:
        img = cv2.imread(os.path.join(path, f), cv2.IMREAD_UNCHANGED)
        img_array = np.array(img)
        arrays.append((img_array.shape, img_array))

    return arrays, png_files


#taken from https://stackoverflow.com/questions/48809433/read-pfm-format-in-python
def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:

        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale

def import_unimatch_info(path, ext):
    npy_files = [f for f in sorted(os.listdir(path)) if f.endswith(ext)]
    arrays = [read_pfm(os.path.join(path, f)) for f in npy_files]
    return arrays, npy_files

def print_and_save_error(gt_depth, pred_depth, pred_output_path, pred_filename, i):
        rmse = np.sqrt(((gt_depth - pred_depth) ** 2).mean())

        predicted_abs_diff = np.abs(gt_depth - pred_depth)
        predicted_diff_img = np.uint16(predicted_abs_diff * 256)
        save_path = os.path.join(pred_output_path, f'{" ".join(pred_filename.split(".")[:-1])}.png')
        cv2.imwrite(save_path, predicted_diff_img)
        return rmse, predicted_abs_diff, predicted_diff_img

def comparing_methods():
    f = 721 #px
    base_length = 0.54 #pm

    calculate_unimatch = False
    calculate_mono = False

    pred_path = "fileoutput/NPY_Depth4"
    unimatch_path = "fileoutput/UNIMATCH"
    pred_output_path = "fileoutput/monodepth_out4"
    unimatch_output_path = "fileoutput/unimatch_out4"
    monodepth_histogram_output_path = "fileoutput/hist4_monodepth"
    unimatch_histogram_output_path = "fileoutput/hist4_unimatch"

    if calculate_unimatch:
        shell_script_path = './gmstereo_demo.sh'
        os.system(f'bash {shell_script_path}')

    if calculate_mono:
        mono_settings = Settings1(image_path='data_ass3/Task4/rectified_images/image_2', output_depth='NPY_Depth4'
                                  , output_disp='NPY_Disp4', output_img='IMG4')
        predict_depth(mono_settings)


    gt_imgs, png_files = import_gt_info("data_ass3/Task4/GT_disparities/disp_noc_0/")
    pred_depths, pred_files = import_depth_info(pred_path, "npy")
    unimatch_disp, unimatch_files = import_unimatch_info(unimatch_path, "pfm")
    #skipping last img for now due to different dimensions
    for i in range(len(png_files)):
        # gt_img = gt_imgs[i]
        # gt_height, gt_width = gt_img.shape[:2]
        #gt are stored as shape and array tuples since last one has different dimensions
        gt_img = gt_imgs[i][1]
        gt_height, gt_width = gt_imgs[i][0]

        mask = np.array(gt_img > 0).reshape((gt_height, gt_width))
        gt_disp = ((gt_img.astype(np.float32) / 256.0))

        ones = np.ones(gt_disp.shape)
        gt_disp = np.divide(ones, gt_disp, out=np.zeros_like(ones), where=gt_disp>0)
        gt_depth = (f * base_length) * gt_disp
        
        unimatch_depth = ((f * base_length) / unimatch_disp[i])
        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

        far_filter_mask = (gt_depth < 120).reshape((gt_height, gt_width))
        gt_depth *=  far_filter_mask
        pred_depth *= mask
        pred_depth *= far_filter_mask
        unimatch_depth *= mask
        unimatch_depth *= far_filter_mask

        mono_rmse, predicted_abs_diff, _ = print_and_save_error(gt_depth, pred_depth, pred_output_path, pred_files[i], i)
        unimatch_rmse, unimatch_abs_diff, _ = print_and_save_error(gt_depth, unimatch_depth, unimatch_output_path, unimatch_files[i], i)
        print(f'(Monodepth) \t {i + 1}. {" ".join(pred_files[i].split(".")[:-1])} : {mono_rmse}')
        print(f'(Unimatch) \t {i + 1}. {" ".join(unimatch_files[i].split(".")[:-1])} : {unimatch_rmse}')

        pred_mask = (predicted_abs_diff > 0) & (predicted_abs_diff <= 10)
        unimatch_mask = (unimatch_abs_diff > 0) & (unimatch_abs_diff <= 10)

        fig = plt.figure()
        predicted_abs_diff = predicted_abs_diff[pred_mask]
        bins = np.arange(min(predicted_abs_diff), max(predicted_abs_diff) + 0.1, 0.1)
        plt.hist(predicted_abs_diff, bins=bins, edgecolor="black")
        plt.title("Monodepth Error Histogram")
        plt.xlabel("Error [m]")
        plt.ylabel("Total")
        mono_fig_path = os.path.join(monodepth_histogram_output_path,f'{" ".join(pred_files[i].split(".")[:-1])}_hist.png')
        fig.savefig(mono_fig_path)

        plt.clf()
        fig = plt.figure()
        unimatch_abs_diff = unimatch_abs_diff[unimatch_mask]
        bins = np.arange(min(unimatch_abs_diff), max(unimatch_abs_diff) + 0.1, 0.1)
        plt.hist(unimatch_abs_diff, bins=100, edgecolor="black")
        plt.xlabel("Error [m]")
        plt.ylabel("Total")
        plt.title("Unimatch Error Histogram")
        unimatch_fig_path = os.path.join(unimatch_histogram_output_path,f'{" ".join(unimatch_files[i].split(".")[:-1])}_hist.png')
        fig.savefig(unimatch_fig_path)
        plt.clf()



if __name__ == "__main__":
    comparing_methods()