import numpy as np
import torch
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import random

from utils.utils import cvtColor, preprocess_input_mel,preprocess_input_video


class DiffusionDataset(Dataset):
    def __init__(self, annotation_lines, input_shape):
        super(DiffusionDataset, self).__init__()

        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        num=0
        videol=[]
        musicl=[]
        f_v_name=self.annotation_lines[index].split(',')[0]+'.npy'
        #f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Tensor_Video',f_v_name)
        f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Video_Clip',f_v_name)
        
        video=torch.tensor(np.load(f_v_root).squeeze())
        
        f_m_name=self.annotation_lines[index].split(',')[0]+'.npy'
        f_m_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Data_music_mel',f_m_name)
        
        music_O=torch.tensor(np.load(f_m_root).squeeze())
        
        len1=music_O.shape[1]//80
        len2=video.size(0)
        num= random.randint(0,min(len1,len2)-4)
        
        for i in range(4):
            j = num+i
            videol.append(video[j])
            musicl.append(music_O[:,80*j:80*(j+1)])
        video = torch.stack(videol)
        music = torch.stack(musicl)
        
        
        
        music   = preprocess_input_mel(music)
        video   = preprocess_input_video(video)
        

        return music,video

def Diffusion_dataset_collate(batch):
    musics = []
    videos = []
    for music,video in batch:
        musics.append(music)
        videos.append(video)
        
        
    musics = torch.stack(musics)
    videos = torch.stack(videos)
    
    
    return musics,videos
