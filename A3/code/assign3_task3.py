import os
from assign3_task1 import Settings1, predict_depth
from assign3_task2 import Settings2, error_eval



print('Loading and predicing with fine tuned model:')
settings = Settings1(model_name='finetune', output_depth='NPY_DepthFT', output_img='IMGFT', output_disp='NPY_DispFT')
predict_depth(settings)


print('Before:')
settings = Settings2()
error_eval(settings)

print('After:')
settings = Settings2(prediction_path='fileoutput/NPY_DepthFT')
error_eval(settings)