from ddpm_video_DDIM_MMV_T import Diffusion
from utils.utils import preprocess_input_video,preprocess_input_mel
import os
import torch
import numpy as np



if __name__ == "__main__":
    
    def preprocess_music(x):
    
        x -= 0.5
        x /= 0.5
        return x
    def postprocess_output_mel(x):
        x *= 0.5
        x += 0.5
        x =13.7*x-11.5
        return x
    
    save_path = "/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Aistt_DDIM_input/gJB_sBM_c01_d07_mJB1_ch07_video_music_ddim0513_r10_MMVTN_10.npy"

    #ddpm = Diffusion('F:/Code_save/diffusion_model_last_epoch_weights.pth')
    ddpm = Diffusion()
        
    f_v_name='gJB_sBM_c01_d07_mJB1_ch07.npy'#视频特征向量
    
    f_m_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Music_mel',f_v_name)
    
    
    f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Video_Clip',f_v_name)
    #video=torch.tensor(np.load(f_v_root)).cuda()[0].unsqueeze(0)#单帧（1，512）
    video = np.load(f_v_root)#视频归一化
    video=preprocess_input_video(torch.tensor(video).cuda()[:,:]).float()#全部帧
    number=video.size(0)
    

# 创建一个形状为 (3, 4) 的张量，填充标准正态分布的随机数
    x = torch.rand(1,1,80,80)
    music=preprocess_music(x.cuda())
    
 

    print("Generate_1x1_image")
    
    
    music_G=[]
    for i in range(number):
        if i==0:
            g = ddpm.generate_1x1_image(music,video[0].unsqueeze(0).unsqueeze(1))
            print(g.shape)
            music_G.append(g)
        else:
            g = ddpm.generate_1x1_image(music_G[i-1].cuda(),video[i].unsqueeze(0).unsqueeze(1))
            music_G.append(g)
            
    list1=[]
    for i in music_G:
        test_images = postprocess_output_mel(i.cpu().data.numpy()).squeeze()
        list1.append(test_images)
    test_images = np.concatenate(list1, axis=1)


    np.save(save_path, test_images)