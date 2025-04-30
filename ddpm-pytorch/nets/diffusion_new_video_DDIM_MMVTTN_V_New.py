import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy
from nets.denoising_VSM import generalized_steps
import tqdm


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class EMA():
    def __init__(self, decay):
        self.decay = decay
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)

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


class GaussianDiffusion(nn.Module):
    def __init__(
        self, model, img_size, img_channels, num_classes=None, betas=[], loss_type="l2", ema_decay=0.9999, ema_start=2000, ema_update_rate=1,
    ):
        super().__init__()
        self.model      = model
        self.ema_model  = deepcopy(model)

        self.ema                = EMA(ema_decay)
        self.ema_decay          = ema_decay
        self.ema_start          = ema_start
        self.ema_update_rate    = ema_update_rate
        self.step               = 0

        self.img_size       = img_size
        self.img_channels   = img_channels
        self.num_classes    = num_classes

        # l1或者l2损失
        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type      = loss_type
       
        
        #--------------------------------------------------------------DDIM修改---------------------
        betas = get_beta_schedule(
            beta_schedule='linear',
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        )
        betas = self.betas = torch.from_numpy(betas).float()#.to(self.device)
        self.num_timesteps = betas.shape[0]

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    def sample(self,batch_size, device,t, y=None, use_ema=True,noise=None,music_f=None,video=None):
        

        #self.model.eval()

        img_id = 10000#len(glob.glob(f"{self.args.image_folder}/*"))
        #print(f"starting from image {img_id}")
        total_n_samples = 50000
        #m = total_n_samples - img_id
        m = 64
        n_rounds = m // 64
        timesteps = t

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                
                x = noise
                x = x.repeat(batch_size,1,1,1)

                x = self.sample_image(x, self.ema_model,timesteps,music_f,video)
                #x = inverse_data_transform(config, x)
        return x
    def sample_gus(self,music,t, y=None, use_ema=True,video=None):
        

        #self.model.eval()

        img_id = 10000#len(glob.glob(f"{self.args.image_folder}/*"))
        #print(f"starting from image {img_id}")
        total_n_samples = 50000
        #m = total_n_samples - img_id
        m = 128*1
        n_rounds = m // 64
        timesteps = t

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                
                

                x = self.sample_image(music.cuda(), self.ema_model,timesteps,video)
                #x = inverse_data_transform(config, x)
        return x
    def sample_image(self, x, model, timesteps=1000,music_f=None,video = None,last=True):
        

        
        
        skip = self.num_timesteps // timesteps
        seq = range(0, self.num_timesteps, skip)

        xs = generalized_steps(x, seq, model, self.betas, music_f,video,eta=0.0)
        x = xs
        
        if last:
            x = x[0][-1]
        return x

    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t,  x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )   

    def get_losses(self, x,x_f,y,video):
        # x, noise [batch_size, 3, 64, 64]
        device = x.device
        n = x.size(0)
        e = torch.randn_like(x)
        
        b = self.betas.to(device)
        
        # antithetic sampling
        t = torch.randint(
            low=0, high=self.num_timesteps, size=(n // 2 + 1,)
        ).to(device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        
        
        
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        
        
        
        perturbed_x = x * a.sqrt() + e * (1.0 - a).sqrt()
        
        
        perturbed_xf = None
        
        estimated_noise = self.model(perturbed_x, perturbed_xf,t.float(), y,video)
        

        if self.loss_type == "l1":
            loss1 = F.l1_loss(estimated_noise, e)
            #loss2 = F.l1_loss(video_emb, x)
            loss = loss1#+0.00001*loss2
        elif self.loss_type == "l2":
            loss1 = F.mse_loss(estimated_noise, e)
            #loss2 = F.mse_loss(video_emb, x)
            loss = loss1#+0.00001*loss2
        return loss

    def forward(self, x, x_f,y=None,video=None):
        b, c, h, w  = x.shape
        device      = x.device

        
        #t = torch.randint(0, self.num_timesteps, (b,), device=device)#self.num_timesteps  = len(betas)
        return self.get_losses(x, x_f,y,video)

def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)
