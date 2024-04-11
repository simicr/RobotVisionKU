#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0 python3 unimatch/main_stereo.py \
--inference_dir unimatch/demo/stereo-middlebury \
--inference_size 1024 1536 \
--output_path fileoutput/disparity_picture.png \
--resume pretrained/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3




