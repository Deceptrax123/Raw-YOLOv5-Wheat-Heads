import torch
from PIL import Image
import torchvision 
import numpy as np 
import torchvision.transforms.v2 as T 
from torch.utils.data import Dataset

class Wheat_Head_Dataset(Dataset):
    def __init__(self,paths):
        self.paths=paths

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index):
        path=self.paths[index]
        sample=Image.open("/Volumes/T7 Shield/Smudge/Datasets/Wheat_Head_detection/test/"+path)

        transform=T.Compose([T.ToImageTensor()])
        sample_tensor=transform(sample)

        mean_image=torch.mean(sample_tensor,[1,2])
        std_image=torch.std(sample_tensor,[1,2])

        normalize=T.Normalize(mean=mean_image,std=std_image)
        img_normalized=normalize(sample_tensor)

        return img_normalized