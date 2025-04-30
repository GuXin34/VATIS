"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util2, logger
from cm.script_util2 import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion2 import karras_sample


def main():
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
    #list1=['gMH_sBM_c01_d23_mMH0_ch07','gJB_sBM_c01_d07_mJB1_ch07','gMH_sBM_c03_d22_mMH1_ch08',
           #'gPO_sBM_c01_d10_mPO2_ch10','gPO_sBM_c02_d11_mPO0_ch09','gWA_sFM_c02_d25_mWA3_ch04']
    list1=['gWA_sBM_c02_d25_mWA2_ch10','gWA_sBM_c09_d25_mWA3_ch10']#,'gPO_sBM_c03_d11_mPO0_ch09','gMH_sBM_c02_d24_mMH5_ch08',
           #'gLO_sBM_c09_d14_mLO0_ch09','gJS_sBM_c02_d03_mJS2_ch03','gJB_sBM_c09_d07_mJB3_ch06','gHO_sBM_c07_d19_mHO1_ch10',
           #'gBR_sBM_c06_d06_mBR4_ch06','gKR_sFM_c09_d28_mKR1_ch02']
    
    if torch.cuda.is_available():

        torch.cuda.manual_seed(123)
        torch.cuda.manual_seed_all(123)  # 如果使用多GPU环境
        
    
    ddpm = Diffusion()
    
  
    for name in list1:
        seed = random.randint(1,2**31-1)
        save_name = name+'VSMBFS2_LT_2T_'+str(seed)+'.npy'
        save_path = os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Aistt_DDIM_input',save_name)
  
        f_v_name = name+'.npy'#视频特征向量

        f_m_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Music_mel','Average80.npy')


        f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Aistt_Video_Clip',f_v_name)
        #video=torch.tensor(np.load(f_v_root)).cuda()[0].unsqueeze(0)#单帧（1，512）
        video = np.load(f_v_root)#视频归一化
        video=preprocess_input_video(torch.tensor(video)[:,:]).float()#全部帧



    # 创建一个形状为 (3, 4) 的张量，填充标准正态分布的随机数
        number=video.size(0)
        
        torch.manual_seed(seed)
        x = torch.randn(1,1,80,80)
        music=x.cuda()
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #music = noise_music(m,x.cuda())
        noise = music#torch.randn(1,1,80,80).cuda()


        device      = torch.device("cuda")
        e =  music#torch.randn(1,1,80,80).cuda()
        
        def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
            """Constructs the noise schedule of Karras et al. (2022)."""
            ramp = th.linspace(0, 1, n)
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            return append_zero(sigmas).to(device)

        shape = (1,1,80,80)
        sigmas = get_sigmas_karras(steps, args.sigma_min, args.sigma_max, 7.0, device=device)
        x_T = generator.randn(*shape, device=device) * args.sigma_max


        music_G=[]
        for i in range(number):
            if i==0:
                video_s=V_LT(video,0).cuda()
                sample = karras_sample(
                            diffusion,
                            model,
                            (1, 1, args.image_size, args.image_size),
                            steps=args.steps,
                            model_kwargs={},
                            device=dist_util.dev(),
                            clip_denoised=args.clip_denoised,
                            sampler=args.sampler,
                            sigma_min=args.sigma_min,
                            sigma_max=args.sigma_max,
                            s_churn=args.s_churn,
                            s_tmin=args.s_tmin,
                            s_tmax=args.s_tmax,
                            s_noise=args.s_noise,
                            generator=generator,
                            ts=ts,
                            x_T = x_T
                            sigmas = sigmas
                            x_f = x_T,
                            video = video_s.unsqueeze(0),
                        )
                g = sample
                print(g.shape)
                music_G.append(g)
            else:
                video_s=V_LT(video,i).cuda()
                music =music_G[i-1].to(device)#noise_music(music_G[i-1],e)#music_G[i-1].to(device)  
                sample = karras_sample(
                            diffusion,
                            model,
                            (1, 1, args.image_size, args.image_size),
                            steps=args.steps,
                            model_kwargs={},
                            device=dist_util.dev(),
                            clip_denoised=args.clip_denoised,
                            sampler=args.sampler,
                            sigma_min=args.sigma_min,
                            sigma_max=args.sigma_max,
                            s_churn=args.s_churn,
                            s_tmin=args.s_tmin,
                            s_tmax=args.s_tmax,
                            s_noise=args.s_noise,
                            generator=generator,
                            ts=ts,
                            x_T = x_T
                            sigmas = sigmas
                            x_f = music,
                            video = video_s.unsqueeze(0),
                        )
                g = sample
                music_G.append(g)

        list1=[]
        for i in music_G:
            test_images = postprocess_output_mel(i.cpu().data.numpy()).squeeze()
            list1.append(test_images)
        test_images = np.concatenate(list1, axis=1)


        np.save(save_path, test_images)
    args = create_argparser().parse_args()

    dist_util2.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)
    
    
    
    

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        sample = karras_sample(
            diffusion,
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
            x_T = x_T
            sigmas = sigmas
            x_f = ,
            video = ,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
