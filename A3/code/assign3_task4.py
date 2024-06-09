import os
from assign3_task1 import Settings1, predict_depth
from assign3_task2 import Settings2, error_eval




def comparing_methods():

    calculate_unimatch = False
    calculate_mono = True

    if calculate_unimatch:
        shell_script_path = './gmstereo_demo.sh'
        os.system(f'bash {shell_script_path}')

    if calculate_mono:
        mono_settings = Settings1(image_path='data_ass3/Task4/rectified_images/image_2', output_depth='NPY_Depth4'
                                  , output_disp='NPY_Disp4', output_img='IMG4')
        predict_depth(mono_settings)


    

if __name__ == "__main__":
    comparing_methods()