from ddpm_video_DDIM_MMV_T_sameN import Diffusion
#from utils.utils import preprocess_input_video,preprocess_input_mel
import os
import torch
import numpy as np
import torch.nn as nn
import math
import random

if __name__ == "__main__":
    
    def ROPE(vectors, seq_len, d_model):
      """
      对输入的向量进行旋转位置编码。
      
      参数:
      vectors -- 输入的向量，形状为 (seq_len, d_model)。
      seq_len -- 序列的长度。
      d_model -- 向量的维度，必须是偶数。
      
      返回:
      encoded_vectors -- 旋转编码后的向量。
      """
      # 确保 d_model 是偶数
      assert d_model % 2 == 0, "d_model must be even"
      
      # 初始化角度数组
      angles = torch.arange(seq_len, device=vectors.device).unsqueeze(-1) * \
               torch.arange(d_model, device=vectors.device) / d_model
      angles = angles.view(seq_len, d_model)
      
      # 计算正弦和余弦值
      sin_angles = torch.sin(angles)
      cos_angles = torch.cos(angles)
      
      # 分离向量为两半
      s = vectors[:, 0::2]
      t = vectors[:, 1::2]
      
      # 旋转编码
      sl = s.size(-1)  # 获取 s 的第二维度大小
      
      tl = t.size(-1)  # 获取 t 的第二维度大小
    
      
      # 扩展正弦和余弦张量以匹配 s 和 t 的形状
      sin_angles = sin_angles[:, :sl]
      cos_angles = cos_angles[:, :tl]
      
      # 执行旋转编码
      encoded_vectors1 = t * cos_angles - s * sin_angles
      encoded_vectors2 = t * sin_angles + s * cos_angles
      
      # 连接编码后的向量
      encoded_vectors = torch.cat([encoded_vectors1, encoded_vectors2], dim=1)
      
      return encoded_vectors
      
    def preprocess_input_video(x):
        x =(x+0.5984243)/1.32733343
        x -= 0.5
        x /= 0.5
        return x
    def MM_melV2_out(x):
        x *= 0.5
        x += 0.5
        x =11.6739689*x-7.8435462
        return x
    def postprocess_output_mel(x):
        means = np.load('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/MVED_Music_melV2_Means.npy')
        std_devs =np.load('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/MVED_Music_melV2_Std.npy')
        x =x* std_devs[:,np.newaxis] + means[:,np.newaxis]
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
        t_f = torch.randint(low=10, high=11, size=(n // 2 + 1,)).to(device)

        t_f = torch.cat([t_f, t_f], dim=0)[:n]
       
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
   
    def V_LT(video,n):
        
        video_list=[]
        number=video.size(0)
        indices = torch.arange(number)  # 获取所有行的索引
        video_list = video[indices != n]
        distances = calculate_distances(n, number)
        weights = softmax(distances)
        weights_tensor = torch.tensor(weights).view(number-1, 1)
        weighted_tensor = (video_list * weights_tensor.expand(-1, video.size(1))).sum(dim=0, keepdim=True)

        video_LT = 0.5*(video[n]+weighted_tensor).to(video[n].dtype)#
        
        return video_LT
    
    def read_lines_to_list(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            # 使用列表推导式读取所有行到列表中
            line_list = [line.strip().split(',')[0] for line in file]
        return line_list


    file_path = '/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/MVED_Test_1.txt'
    list3 = read_lines_to_list(file_path)
    
    list2=['gWA_sBM_c02_d25_mWA2_ch10','gWA_sBM_c09_d25_mWA3_ch10','gPO_sBM_c03_d11_mPO0_ch09','gMH_sBM_c02_d24_mMH5_ch08','gLO_sBM_c09_d14_mLO0_ch09','gJS_sBM_c02_d03_mJS2_ch03','gJB_sBM_c09_d07_mJB3_ch06','gHO_sBM_c07_d19_mHO1_ch10','gBR_sBM_c06_d06_mBR4_ch06','gKR_sFM_c09_d28_mKR1_ch02']
#'gWA_sBM_c02_d25_mWA2_ch10','gWA_sBM_c09_d25_mWA3_ch10','gPO_sBM_c03_d11_mPO0_ch09','gMH_sBM_c02_d24_mMH5_ch08','gLO_sBM_c09_d14_mLO0_ch09','gJS_sBM_c02_d03_mJS2_ch03','gJB_sBM_c09_d07_mJB3_ch06','gHO_sBM_c07_d19_mHO1_ch10','gBR_sBM_c06_d06_mBR4_ch06','gKR_sFM_c09_d28_mKR1_ch02']
#'gWA_sBM_c02_d25_mWA2_ch10','gWA_sBM_c09_d25_mWA3_ch10','gPO_sBM_c03_d11_mPO0_ch09','gMH_sBM_c02_d24_mMH5_ch08','gLO_sBM_c09_d14_mLO0_ch09','gJS_sBM_c02_d03_mJS2_ch03','gJB_sBM_c09_d07_mJB3_ch06','gHO_sBM_c07_d19_mHO1_ch10','gBR_sBM_c06_d06_mBR4_ch06','gKR_sFM_c09_d28_mKR1_ch02']
    list4=['gMH_sFM_c09_d22_mMH4_ch05','gPO_sBM_c03_d10_mPO0_ch06']#,'gPO_sBM_c03_d10_mPO2_ch01','gPO_sBM_c05_d12_mPO5_ch07','gPO_sBM_c08_d12_mPO4_ch09','gPO_sBM_c09_d11_mPO4_ch10','gPO_sFM_c04_d10_mPO2_ch03','gPO_sFM_c04_d12_mPO0_ch15','gWA_sBM_c04_d25_mWA3_ch04','gWA_sBM_c05_d25_mWA0_ch10','gWA_sBM_c08_d27_mWA2_ch10','gWA_sBM_c09_d26_mWA4_ch05','gWA_sBM_c09_d27_mWA3_ch08','gWA_sFM_c05_d27_mWA1_ch16','gWA_sFM_c09_d25_mWA5_ch06',]      
    
    if torch.cuda.is_available():

        torch.cuda.manual_seed(123)
        torch.cuda.manual_seed_all(123)  # 如果使用多GPU环境
    
    NUMBER=1
    for N in range(NUMBER):
        
        save_path_N=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Aistt_DDIM_input',str(N+1)) 
        if not os.path.exists(save_path_N):
            os.makedirs(save_path_N)
    
        for name in list3:

            seed = random.randint(1,2**31-1)
            save_name = name+'VMBFS23_New_NorMM_'+str(seed)+'.npy'

            save_path = os.path.join(save_path_N,save_name)

            ddpm = Diffusion()

            f_v_name = name+'.npy'#视频特征向量

            #f_m_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Music_melV2',f_v_name)


            f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/MVED_Video_MetaClip',f_v_name)
            #video=torch.tensor(np.load(f_v_root)).cuda()[0].unsqueeze(0)#单帧（1，512）
            video = np.load(f_v_root)#视频归一化
            video=preprocess_input_video(torch.tensor(video)[:,:]).float()#全部帧

            video =  ROPE(video,video.size(0),512)

            #m = np.load(f_m_root)
            #m =preprocess_input_mel(torch.tensor(m)).float()
            #print(m.shape)

        # 创建一个形状为 (3, 4) 的张量，填充标准正态分布的随机数
            number=video.size(0)

            torch.manual_seed(seed)
            x = torch.randn(1,1,80,88)
            music=x.cuda()#m[:,:,:80].unsqueeze(0).cuda()
            device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            #music = noise_music(m,x.cuda())
            noise = music#torch.randn(1,1,80,88).cuda()#music


            device      = torch.device("cuda")
            e =  music#torch.randn(1,1,80,88).cuda()


            music_G=[]
            for i in range(number):
                if i==0:
                    #video_s=V_LT(video,0).cuda()
                    music = torch.zeros(1,1,80,88).to(device)
                    music = noise_music(music,e)
                    video_INIT = torch.mean(video,dim=0).unsqueeze(0).cuda()
                    g = ddpm.generate_1x1_image(noise,music,video_INIT.unsqueeze(0))
                    #g = ddpm.generate_1x1_image(noise,g.to(device),video_s.unsqueeze(0))
                    print(g.shape)
                    music_G.append(g)
                else:
                    video_s=V_LT(video,i).cuda()
                    music =noise_music(music_G[i-1],e)#music_G[i-1].to(device)                                                                                                                                                                                                                                                                                                                                                                                
                    g = ddpm.generate_1x1_image(noise,music,video_s.unsqueeze(0))
                    music_G.append(g)

            list1=[]
            list_S=[]
            list_E=[]
            for i in range(len(music_G)):
                test_images = postprocess_output_mel(MM_melV2_out(music_G[i].cpu().data.numpy())).squeeze()
                if i==0:
                    print(test_images.shape)
                    list1.append(test_images[:,:86])
                    list_E.append(test_images[:,-2:])
                else:
                    list_S.append(test_images[:,:2])
                    A=(list_E[i-1]+list_S[i-1])/2
                    list1.append(A)
                    list1.append(test_images[:,2:-2])
                    list_E.append(test_images[:,-2:])


            test_images = np.concatenate(list1, axis=1)


            np.save(save_path, test_images)