from ddpm_video_DDIM_MMV_T_sameN import Diffusion
from utils.utils import preprocess_input_video,preprocess_input_mel
import os
import torch
import numpy as np
import random


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
    
    def calculate_distances(n, m):
    # 计算从0到m的每个整数到n的距离，排除n自身
        if n is None or m is None:
            raise ValueError("The values of n and m must not be None")
        return [abs(i - n) for i in range(m) if i != n]

    def softmax(logits):
    # 将logits转换为权重
        exp_logits = np.exp(logits - np.max(logits))  # 稳定softmax计算
        return exp_logits / exp_logits.sum(axis=0, keepdims=True)
   
    def V_LT(video,i):
        
        videol=[]
        for j in range(i,i+2):
            video_list=[]
            number=video.size(0)
            indices = torch.arange(number)  # 获取所有行的索引
            video_list = video[indices != j]
            distances = calculate_distances(j, number)
            weights = softmax(distances)
            weights_tensor = torch.tensor(weights).view(number-1, 1)
            weighted_tensor = (video_list * weights_tensor.expand(-1, video.size(1))).sum(dim=0, keepdim=True)

            video_1 = 0.5*(video[j]+weighted_tensor).to(video[j].dtype)#
            videol.append(video_1)
        
        video_LT = torch.stack(videol).squeeze()
        
        
        return video_LT
    #list1=['gMH_sBM_c01_d23_mMH0_ch07','gJB_sBM_c01_d07_mJB1_ch07','gMH_sBM_c03_d22_mMH1_ch08',
           #'gPO_sBM_c01_d10_mPO2_ch10','gPO_sBM_c02_d11_mPO0_ch09','gWA_sFM_c02_d25_mWA3_ch04']
    #list1=['gMH_sBM_c01_d23_mMH0_ch07','gJB_sBM_c01_d07_mJB1_ch07','gPO_sBM_c01_d10_mPO2_ch10']
    #list_seed=[675248436,675248436,675248436]
    j=0
    
        
    list1=['gWA_sBM_c02_d25_mWA2_ch10','gWA_sBM_c09_d25_mWA3_ch10','gPO_sBM_c03_d11_mPO0_ch09','gMH_sBM_c02_d24_mMH5_ch08',
           'gLO_sBM_c09_d14_mLO0_ch09','gJS_sBM_c02_d03_mJS2_ch03','gJB_sBM_c09_d07_mJB3_ch06','gHO_sBM_c07_d19_mHO1_ch10',
           'gBR_sBM_c06_d06_mBR4_ch06','gKR_sFM_c09_d28_mKR1_ch02']
    
    if torch.cuda.is_available():

        torch.cuda.manual_seed(123)
        torch.cuda.manual_seed_all(123)  # 如果使用多GPU环境
    
    for name in list1:
        
        seed = random.randint(1,2**31-1)
        save_name = name+'VSMBFS5BO_LT_2T_'+str(seed)+'.npy'
        
        save_path = os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Aistt_DDIM_input',save_name)

        ddpm = Diffusion()

        f_v_name = name+'.npy'#视频特征向量

        f_m_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Music_mel',f_v_name)


        f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Video_Clip',f_v_name)
        #video=torch.tensor(np.load(f_v_root)).cuda()[0].unsqueeze(0)#单帧（1，512）
        video = np.load(f_v_root)#视频归一化
        video=preprocess_input_video(torch.tensor(video)[:,:]).float()#全部帧
        
        
    # 创建一个形状为 (3, 4) 的张量，填充标准正态分布的随机数
        number=video.size(0)
        
        torch.manual_seed(seed)
        x = torch.randn(1,1,80,160)
        music=x.cuda()
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        noise = music#torch.randn(1,1,80,80).cuda()



        device      = torch.device("cuda")
        e =  music#torch.randn(1,1,80,80).cuda()


        music_G=[]
        for i in range(number-2):
            video_s=V_LT(video,i).cuda()
            if i==0:
                g = ddpm.generate_1x1_image(noise,music,video_s.unsqueeze(0))
                print(g.shape)
                music_G.append(g[:,:,:,:80])
                music_G.append(g[:,:,:,80:160])
            
            else:
                music = torch.cat((music_G[-2],music_G[-1]),dim=-1).to(device)
                g = ddpm.generate_1x1_image(noise,music,video_s.unsqueeze(0))
                music_G.pop(-1)
                music_G.append(g[:,:,:,:80])
                music_G.append(g[:,:,:,80:160])

        list2=[]
        for i in music_G:
            test_images = postprocess_output_mel(i.cpu().data.numpy()).squeeze()
            list2.append(test_images)
            
            print(test_images.shape)
        test_images = np.concatenate(list2, axis=1)


        np.save(save_path, test_images)