from ddpm_video_DDIM_MMV_T_sameN import Diffusion
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
    def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        if beta_schedule == "quad":
            betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
            )
        elif beta_schedule == "linear":
            betas = np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / np.linspace(
                num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_diffusion_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_diffusion_timesteps,)
        return betas
    
    def noise_music(x,e):
        n=x.size(0)
        betas = get_beta_schedule(
            beta_schedule='linear',
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,)
        b = torch.from_numpy(betas).float().to(device)
        t_f = torch.randint(low=998, high=1000, size=(n // 2 + 1,)).to(device)

        t_f = torch.cat([t_f, 1000 - t_f - 1], dim=0)[:n]

       
        a_f = (1-b).cumprod(dim=0).index_select(0, t_f).view(-1, 1, 1, 1)

        music = x.to(device) * a_f.sqrt() + e * (1.0 - a_f).sqrt()
        
        return music
    
   
    
    #list1=['gMH_sBM_c01_d23_mMH0_ch07','gJB_sBM_c01_d07_mJB1_ch07','gMH_sBM_c03_d22_mMH1_ch08',
           #'gPO_sBM_c01_d10_mPO2_ch10','gPO_sBM_c02_d11_mPO0_ch09','gWA_sFM_c02_d25_mWA3_ch04']
    list1=['gMH_sBM_c01_d23_mMH0_ch07','gJB_sBM_c01_d07_mJB1_ch07']
    device      = torch.device('cpu' )
    for name in list1:
        
        save_name = name+'_r10_r12_MMVTTN_C1-0_2T_01.npy'
        save_path = os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Aistt_DDIM_input',save_name)

        ddpm = Diffusion()

        f_v_name = name+'.npy'#视频特征向量

        f_m_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Music_mel',f_v_name)


        f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Video_Clip',f_v_name)
        #video=torch.tensor(np.load(f_v_root)).cuda()[0].unsqueeze(0)#单帧（1，512）
        video = np.load(f_v_root)#视频归一化
        video=preprocess_input_video(torch.tensor(video).to(device)[:,:]).float()#全部帧
        number=video.size(0)

        
    # 创建一个形状为 (3, 4) 的张量，填充标准正态分布的随机数
        x = torch.randn(1,1,80,80)
        music=x.to(device)


        noise = music#torch.randn(1,1,80,80).to(device)



        e = noise#torch.randn(1,1,80,80).to(device)


        music_G=[]
        for i in range(number):
            if i==0:
                g = ddpm.generate_1x1_image(noise,music,video[0].unsqueeze(0).unsqueeze(1))
                print(g.shape)
                music_G.append(g)
            else:
                music = noise_music(music_G[i-1],e)
                g = ddpm.generate_1x1_image(noise,music,video[i].unsqueeze(0).unsqueeze(1))
                music_G.append(g)

        list1=[]
        for i in music_G:
            test_images = postprocess_output_mel(i.cpu().data.numpy()).squeeze()
            list1.append(test_images)
        test_images = np.concatenate(list1, axis=1)


        np.save(save_path, test_images)