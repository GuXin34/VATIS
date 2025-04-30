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
    
    def calculate_distances(n, m):
    # 计算从0到m的每个整数到n的距离，排除n自身
        if n is None or m is None:
            raise ValueError("The values of n and m must not be None")
        return [abs(i - n) for i in range(m) if i != n]

    def softmax(logits):
    # 将logits转换为权重
        exp_logits = np.exp(logits - np.max(logits))  # 稳定softmax计算
        return exp_logits / exp_logits.sum(axis=0, keepdims=True)
   
    def V_FL(video,n,flag):
        
        if flag is False:
            if n==0:
                a  =video[:2,]
                video_FL = torch.cat((video[n,:].unsqueeze(0),a), axis=0)
            else:
                video_FL = video[n-1:n+2,:]
        else:
            a = video[-2:,:]
            video_FL = torch.cat((a, video[n,:].unsqueeze(0)), axis=0)
            
        return video_FL
    #list1=['gMH_sBM_c01_d23_mMH0_ch07','gJB_sBM_c01_d07_mJB1_ch07','gMH_sBM_c03_d22_mMH1_ch08',
           #'gPO_sBM_c01_d10_mPO2_ch10','gPO_sBM_c02_d11_mPO0_ch09','gWA_sFM_c02_d25_mWA3_ch04']
    list1=['gMH_sBM_c01_d23_mMH0_ch07','gJB_sBM_c01_d07_mJB1_ch07']
    for name in list1:
        
        save_name = name+'VSMBFS4_LT_20T.npy'
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
        x = torch.randn(1,1,80,80)
        music=x.cuda()
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        noise = music#torch.randn(1,1,80,80).cuda()

        device      = torch.device("cuda")
        e =  music#torch.randn(1,1,80,80).cuda()


        music_G=[]
        for i in range(number):
            if i==0:
                video_s=V_FL(video,0,False).cuda()
                g = ddpm.generate_1x1_image(noise,music,video_s.unsqueeze(0))
                print(g.shape)
                music_G.append(g)
                
            elif i==number-1:
                video_s=V_FL(video,i,True).cuda()
                music =music_G[i-1].to(device)#noise_music(music_G[i-1],e)#music_G[i-1].to(device)                                                                                                                                                                                                                                                                                                                                                                                
                g = ddpm.generate_1x1_image(noise,music,video_s.unsqueeze(0))
                music_G.append(g)
                
            else:
                video_s=V_FL(video,i,False).cuda()
                music =music_G[i-1].to(device)#noise_music(music_G[i-1],e)#music_G[i-1].to(device)                                                                                                                                                                                                                                                                                                                                                                                
                g = ddpm.generate_1x1_image(noise,music,video_s.unsqueeze(0))
                music_G.append(g)

        list1=[]
        for i in music_G:
            test_images = postprocess_output_mel(i.cpu().data.numpy()).squeeze()
            list1.append(test_images)
        test_images = np.concatenate(list1, axis=1)

        np.save(save_path, test_images)