from ddpm_video_DDIM_MMV import Diffusion
from utils.utils import preprocess_input_video,preprocess_input_mel
import os
import torch
import numpy as np



if __name__ == "__main__":
    
    save_path = "/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Aistt_DDIM_input/gMH_sBM_c01_d23_mMH0_ch07_video_music_ddim0430_r10_MMV_20.npy"

    #ddpm = Diffusion('F:/Code_save/diffusion_model_last_epoch_weights.pth')
    ddpm = Diffusion()
        
    f_v_name='gMH_sBM_c01_d23_mMH0_ch07.npy'#视频特征向量
    
    f_m_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Music_mel',f_v_name)
    
    
    f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Video_Clip',f_v_name)
    #video=torch.tensor(np.load(f_v_root)).cuda()[0].unsqueeze(0)#单帧（1，512）
    video = np.load(f_v_root)#视频归一化
    video=preprocess_input_video(torch.tensor(video).cuda()[1:,:]).float()#全部帧
    
    music_O = torch.tensor(np.load(f_m_root)).squeeze()
    number=video.size(0)
    musicl=[]
    for i in range(number):
        musicl.append(music_O[:,80*i:80*(i+1)])

    music = torch.stack(musicl)
    music   = preprocess_input_mel(music.cuda())
 
    
    print(video.unsqueeze(1).shape)
    print(music.unsqueeze(1).shape)
    

    print("Generate_1x1_image")
    
        
    ddpm.generate_1x1_image(save_path,music.unsqueeze(1),video.unsqueeze(1))

    original_array=np.load(save_path)
    