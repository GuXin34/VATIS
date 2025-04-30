import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from nets import (GaussianDiffusion, UNet, generate_cosine_schedule,
                  generate_linear_schedule)
from utils.utils import show_config, postprocess_output_mel

class Diffusion(object):
    _defaults = {
        #-----------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #-----------------------------------------------#
        "model_path"        : '/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/logs1/0425Diffusion_Epoch260-GLoss0.0124.pth',
        #-----------------------------------------------#
        #   卷积通道数的设置
        #-----------------------------------------------#
        "channel"           : 16,
        #-----------------------------------------------#
        #   输入图像大小的设置
        #-----------------------------------------------#
        "input_shape"       : (80,80),
        #-----------------------------------------------#
        #   betas相关参数
        #-----------------------------------------------#
        "schedule"          : "linear",
        "num_timesteps"     : 100,
        "schedule_low"      : 1e-4,
        "schedule_high"     : 0.02,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,#True,
    }

    #---------------------------------------------------#
    #   初始化Diffusion
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
            self._defaults[name] = value 
        self.generate()

        show_config(**self._defaults)

    def generate(self):
        #----------------------------------------#
        #   创建Diffusion模型
        #----------------------------------------#
        if self.schedule == "cosine":
            betas = generate_cosine_schedule(self.num_timesteps)
        else:
            betas = generate_linear_schedule(
                self.num_timesteps,
                self.schedule_low * 1000 / self.num_timesteps,
                self.schedule_high * 1000 / self.num_timesteps,
            )
            
        self.net    = GaussianDiffusion(UNet(1, self.channel), self.input_shape, 1)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#'cuda' if torch.cuda.is_available() else 
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Diffusion5x5的图片
    #---------------------------------------------------#
    def generate_5x5_image(self, save_path):
        with torch.no_grad():
            randn_in    = torch.randn((1, 1)).cuda() if self.cuda else torch.randn((1, 1))

            test_images = self.net.sample(25, randn_in.device)

            size_figure_grid = 5
            fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
            for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

            for k in range(5*5):
                i = k // 5
                j = k % 5
                ax[i, j].cla()
                ax[i, j].imshow(np.uint8(postprocess_output(test_images[k].cpu().data.numpy().transpose(1, 2, 0))))

            label = 'predict_5x5_results'
            fig.text(0.5, 0.04, label, ha='center')
            plt.savefig(save_path)

    #---------------------------------------------------#
    #   Diffusion1x1的图片
    #---------------------------------------------------#
    def generate_1x1_image(self, save_path='',music_f=None,video=None):
        with torch.no_grad():
            randn_in    = torch.randn((1, 1)).cuda() if self.cuda else torch.randn((1, 1))
            
            test_images = self.net.sample(video.size(0), randn_in.device, self.num_timesteps,None, use_ema=False,music_f=music_f,video=video)#最初video.size(0)是1
            
            
            
                
            #test_images = postprocess_output(test_images[0].cpu().data.numpy().transpose(1, 2, 0))
            test_images = postprocess_output_mel(test_images.cpu().data.numpy()).squeeze()
            #test_images = test_images.cpu().data.numpy().squeeze()
            print(test_images.shape)
            #-----------------------记得加上.cpu().data.numpy()
            
            
            list1=[]
            for i in test_images:
                list1.append(i)
            test_images = np.concatenate(list1, axis=1)
            
            
            np.save(save_path, test_images)
            print('完成')

            #Image.fromarray(np.uint8(test_images)).save(save_path)