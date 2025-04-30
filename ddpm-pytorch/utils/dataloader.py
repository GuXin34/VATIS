import numpy as np
import torch
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import random

from utils.utils import cvtColor, preprocess_input_encodec


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
        
        f_v_name=self.annotation_lines[index].split(',')[0]+'.npy'
        #f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Tensor_Video',f_v_name)
        f_v_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Video_Clip',f_v_name)
        video=torch.tensor(np.load(f_v_root).squeeze())
        
        f_m_name=self.annotation_lines[index].split(',')[0]+'.npy'
        #f_m_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Data_music_NP',f_m_name)
        f_m_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Data_music_encodec',f_m_name)
        
        music_O=np.load(f_m_root).squeeze()
        
        len1=music_O.shape[0]
        len2=video.size(0)
        number= random.randint(0,min(len1,len2)-1)
        
        music = music_O[number]
        video = video[number]
        
        
        music=np.hstack((music,music[:,-10:]))
        
        
        music=torch.tensor(music)
        music   = preprocess_input_encodec(music)
  
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
