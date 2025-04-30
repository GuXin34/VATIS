from ddpm_video_DDIM_MMV_T_sameN import Diffusion
from utils.utils import preprocess_input_video,preprocess_input_mel
import os
import torch
import numpy as np
import torch.nn as nn
import math
import random

if __name__ == "__main__":
    
    class PositionalEncodingVideo(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super(PositionalEncodingVideo, self).__init__()
            self.d_model = d_model
            # 初始化位置编码矩阵
            pe = torch.zeros(max_len, d_model)

            # 为每个位置和每个维度生成位置编码
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            # 注册位置编码为模型的缓冲区
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            self.register_buffer('pe', pe)

        def forward(self, x):
            # x: [batch_size, N, d_model] 其中N是帧数
            batch_size, N, _ = x.size()
            # 获取对应长度的位置编码
            pe = self.pe[:, :N, :]  # [1, N, d_model]
            # 扩展位置编码的批量维度以匹配x
            pe = pe.expand(batch_size, -1, -1)  # [batch_size, N, d_model]
            # 将位置编码添加到每个帧的特征上
            return x + pe
    
    def MM_melV2_out(x):
        x *= 0.5
        x += 0.5
        x =10.09142302*x-6.46885724
        return x
    def postprocess_output_mel(x):
        means = np.load('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_music_melV2_Means.npy')
        std_devs =np.load('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_music_melV2_Std.npy')
        x =x* (std_devs[:,np.newaxis] + 1e-8) + means[:,np.newaxis]
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
        t_f = torch.randint(low=999, high=1000, size=(n // 2 + 1,)).to(device)

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
   
    def V_LT_POS(video,n):
        
        video_list=[]
        number=video.size(0)
        indices = torch.arange(number)  # 获取所有行的索引
        video_list = video[indices != n]
        distances = calculate_distances(n, number)
        weights = softmax(distances)
        weights_tensor = torch.tensor(weights).view(number-1, 1)
        weighted_tensor = (video_list * weights_tensor.expand(-1, video.size(1))).sum(dim=0, keepdim=True)

        video_LT = 0.5*(video[n]+weighted_tensor).to(video[n].dtype)#
        
        video_n = video_LT.repeat(number, 1)
        video_n =  positional_encoder(video_n.unsqueeze(0)).squeeze()
        video_LT_POS = video_n[n]
        
        return video_LT_POS
    list1=['gWA_sBM_c02_d25_mWA2_ch10','gWA_sBM_c09_d25_mWA3_ch10','gPO_sBM_c03_d11_mPO0_ch09','gMH_sBM_c02_d24_mMH5_ch08','gLO_sBM_c09_d14_mLO0_ch09','gJS_sBM_c02_d03_mJS2_ch03','gJB_sBM_c09_d07_mJB3_ch06','gHO_sBM_c07_d19_mHO1_ch10','gBR_sBM_c06_d06_mBR4_ch06','gKR_sFM_c09_d28_mKR1_ch02','gWA_sBM_c02_d25_mWA2_ch10','gWA_sBM_c09_d25_mWA3_ch10','gPO_sBM_c03_d11_mPO0_ch09','gMH_sBM_c02_d24_mMH5_ch08','gLO_sBM_c09_d14_mLO0_ch09','gJS_sBM_c02_d03_mJS2_ch03','gJB_sBM_c09_d07_mJB3_ch06','gHO_sBM_c07_d19_mHO1_ch10','gBR_sBM_c06_d06_mBR4_ch06','gKR_sFM_c09_d28_mKR1_ch02','gWA_sBM_c02_d25_mWA2_ch10','gWA_sBM_c09_d25_mWA3_ch10','gPO_sBM_c03_d11_mPO0_ch09','gMH_sBM_c02_d24_mMH5_ch08','gLO_sBM_c09_d14_mLO0_ch09','gJS_sBM_c02_d03_mJS2_ch03','gJB_sBM_c09_d07_mJB3_ch06','gHO_sBM_c07_d19_mHO1_ch10','gBR_sBM_c06_d06_mBR4_ch06','gKR_sFM_c09_d28_mKR1_ch02']
           #'gPO_sBM_c01_d10_mPO2_ch10','gPO_sBM_c02_d11_mPO0_ch09','gWA_sFM_c02_d25_mWA3_ch04']
    #list1=['gMH_sBM_c01_d23_mMH0_ch07','gJB_sBM_c01_d07_mJB1_ch07','gPO_sBM_c01_d10_mPO2_ch10']
    
    if torch.cuda.is_available():

        torch.cuda.manual_seed(123)
        torch.cuda.manual_seed_all(123)  # 如果使用多GPU环境

   
    def read_lines_to_list(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            # 使用列表推导式读取所有行到列表中
            line_list = [line.strip()[:-1] for line in file]
        return line_list


    file_path = '/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Test_Aistt.txt'
    list3 = read_lines_to_list(file_path)
    for name in list3:
        
        seed = random.randint(1,2**31-1)
        save_name = name+'VM4-2_POS_LT_2T_'+str(seed)+'.npy'
     
        save_path = os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/10GEN/VM4-2',save_name)

        ddpm = Diffusion()

        f_v_name = name+'.npy'#视频特征向量

        f_m_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Music_melV2',f_v_name)


        f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Video_Clip',f_v_name)
        #video=torch.tensor(np.load(f_v_root)).cuda()[0].unsqueeze(0)#单帧（1，512）
        video = np.load(f_v_root)#视频归一化
        video=preprocess_input_video(torch.tensor(video)[:,:]).float()#全部帧
        
        positional_encoder = PositionalEncodingVideo(512, video.size(0))
        
        
        m = np.load(f_m_root)
        m =preprocess_input_mel(torch.tensor(m)).float()
        #print(m.shape)

    # 创建一个形状为 (3, 4) 的张量，填充标准正态分布的随机数
        number=video.size(0)
        
        torch.manual_seed(seed)
        x = torch.randn(1,1,80,80)
        music=x.cuda()#m[:,:,:80].unsqueeze(0).cuda()
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #music = noise_music(m,x.cuda())
        noise = music#torch.randn(1,1,80,80).cuda()#music


        device      = torch.device("cuda")
        e =  music#torch.randn(1,1,80,80).cuda()


        music_G=[]
        for i in range(number):
            if i==0:
                video_s=V_LT_POS(video,0).cuda()
                g = ddpm.generate_1x1_image(noise,music,video_s.unsqueeze(0))
                print(g.shape)
                music_G.append(g)
            else:
                video_s=V_LT_POS(video,i).cuda()
                music =music_G[i-1].to(device)#noise_music(music_G[i-1],e)#music_G[i-1].to(device)                                                                                                                                                                                                                                                                                                                                                                                
                g = ddpm.generate_1x1_image(noise,music,video_s.unsqueeze(0))
                music_G.append(g)

        list1=[]
        for i in music_G:
            test_images = postprocess_output_mel(MM_melV2_out(i.cpu().data.numpy())).squeeze()
            list1.append(test_images)
        test_images = np.concatenate(list1, axis=1)


        np.save(save_path, test_images)