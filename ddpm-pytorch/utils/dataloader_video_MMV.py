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
        
        f_v_name=self.annotation_lines[index].split(',')[0]+'.npy'
        f_v_root=os.path.join('/home/u202310081000110/Data/Aistt_Video_Clip',f_v_name)
        #f_v_root=os.path.join('F:Data/Video_Clip',f_v_name)
        video=torch.tensor(np.load(f_v_root).squeeze())
        
        f_m_name=self.annotation_lines[index].split(',')[0]+'.npy'
        f_m_root=os.path.join('/home/u202310081000110/Data/Aistt_Music_mel',f_m_name)
        #f_m_root=os.path.join('F:Data/Data_music_mel',f_m_name)
        
        music_O=np.load(f_m_root).squeeze()
        
        len1=music_O.shape[1]//80 - 1
        len2=video.size(0)
        number= random.randint(1,min(len1,len2)-1)
        
        music = music_O[:,80*number:80*(number+1)]
        video = video[number]
        
        music_f = music_O[:,80*(number-1):80*number]
        
        
        music=torch.tensor(music)
        music   = preprocess_input_mel(music)
        music = music.unsqueeze(0)
        
        music_f=torch.tensor(music_f)
        music_f   = preprocess_input_mel(music_f)
        music_f = music_f.unsqueeze(0)
        
        video = preprocess_input_video(video.unsqueeze(0))
        #video = video.unsqueeze(0)

        return music,music_f,video

def Diffusion_dataset_collate(batch):
    musics = []
    musics_f = []
    videos = []
    for music,music_f,video in batch:
        musics.append(music)
        musics_f.append(music_f)
        videos.append(video)
        
        
    musics = torch.stack(musics)
    musics_f = torch.stack(musics_f)
    videos = torch.stack(videos)
    
    
    return musics,musics_f,videos
