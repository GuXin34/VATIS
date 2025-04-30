from ddpm_video import Diffusion
import os
import torch
import numpy as np



if __name__ == "__main__":
    #save_path_5x5 = "results/predict_out/predict_5x5_results.png"
    save_path = "/home/vatis/DataDisk_1/23_vatis_PhD/guxin/input_numpy/100601_video_music_all_tongzao_1020_2.npy"

    #ddpm = Diffusion('F:/Code_save/diffusion_model_last_epoch_weights.pth')
    ddpm = Diffusion()
           
        
    f_v_name='100601.npy'#视频特征向量

    f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Video_Clip',f_v_name)
    video=torch.tensor(np.load(f_v_root)).cuda()
    print(video.shape)
    

    print("Generate_1x1_image")
    
        
    ddpm.generate_1x1_image(save_path,video.unsqueeze(1))

    original_array=np.load(save_path)


