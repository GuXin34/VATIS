
from ddpm_mel_DDIM import Diffusion
from utils.utils import preprocess_input_video
import os
import torch
import numpy as np



if __name__ == "__main__":
    
    save_path = "/home/vatis/DataDisk_1/23_vatis_PhD/guxin/input_numpy_1120/1116_ddim_03.npy"

    #ddpm = Diffusion('F:/Code_save/diffusion_model_last_epoch_weights.pth')
    ddpm = Diffusion()
        

    

    

    print("Generate_1x1_image")
    
        
    ddpm.generate_1x1_image(save_path)

    original_array=np.load(save_path)
    