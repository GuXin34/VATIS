
from ddpm_video_DDIM_4 import Diffusion
from utils.utils import preprocess_input_video
import os
import torch
import numpy as np



if __name__ == "__main__":
    
    save_path = "/home/vatis/DataDisk_1/23_vatis_PhD/guxin/input_numpy_4C/gBR_sBM_c01_d04_mBR0_ch01_04017.npy"

    #ddpm = Diffusion('F:/Code_save/diffusion_model_last_epoch_weights.pth')
    ddpm = Diffusion()
        
    f_v_name='gBR_sBM_c01_d04_mBR0_ch01.npy'#视频特征向量

    f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Video_Clip',f_v_name)
    #video=torch.tensor(np.load(f_v_root)).cuda()[0].unsqueeze(0)#单帧（1，512）
    video = preprocess_input_video(np.load(f_v_root))#视频归一化
    #video = np.load(f_v_root)
    #
    
    num = video.shape[0]
    n = num//4
    video = video[:4*n,:]
    
    video = video.reshape(n,4,-1)
    video=torch.tensor(video).cuda().float()#全部帧
    
  
    

    print("Generate_1x1_image")
    
        
    ddpm.generate_1x1_image(save_path,video)

    original_array=np.load(save_path)
    