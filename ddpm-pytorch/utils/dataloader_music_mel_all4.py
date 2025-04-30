import numpy as np
import torch
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import random

from utils.utils import cvtColor, preprocess_input_mel


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
        
        
        
        f_m_name=self.annotation_lines[index].split(',')[0]+'.npy'
        f_m_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data/Data_music_mel',f_m_name)
        #f_m_root=os.path.join('/home/vatis/DataDisk_1/23_vatis_PhD/guxin/Data_music_encodec',f_m_name)
        #f_m_root=os.path.join('F:Data/Data_music_mel',f_m_name)
        
        music_O=np.load(f_m_root).squeeze()
        
        
        number = int(music_O.shape[1])
        n = np.random.randint(0,number-4*80)
        music1 = music_O[:,n:n+80]
        music2 = music_O[:,n+80:n+2*80]
        music3 = music_O[:,n+2*80:n+3*80]
        music4 = music_O[:,n+3*80:n+4*80]
        music_1 = np.concatenate((music1,music2),axis=1)
        music_2 = np.concatenate((music3,music4),axis=1)
        music = np.concatenate((music_1,music_2),axis=0)
        
      
        music=torch.tensor(music)
        music   = preprocess_input_mel(music)
        music = music.unsqueeze(0)
  
        return music

def Diffusion_dataset_collate(batch):
    musics = []
    
    for music in batch:
        musics.append(music)
        
        
        
    musics = torch.stack(musics)
    
    
    
    return musics
