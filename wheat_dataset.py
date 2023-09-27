import torch 
import torchvision.transforms.v2 as T 
from torch.utils.data import Dataset
import numpy as np 
import pandas as pd 
from PIL import Image 

class WheatDataset(Dataset):
    def __init__(self,image_names,data_dict):
        self.image_names=image_names
        self.data_dict=data_dict

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self,index):

        bounding_box_coords=self.data_dict[self.image_names[index]]
        global_image_path="./train/train/"

        #transforms
        img_to_tensor=T.ToImageTensor()
        
        #get the X
        img=Image.open(global_image_path+self.image_names[index]+".png")
        img_np=np.array(img)
        img_np=img_np.astype(np.float32)

        image_tensor=img_to_tensor(img_np)

        #normalize
        mean_img=torch.mean(image_tensor,[1,2])
        std_img=torch.mean(image_tensor,[1,2])

        normalize_transform=T.Normalize(mean=mean_img,std=std_img)
        image_tensor_normalized=normalize_transform(image_tensor)

        #refine Y
        boxes_string=bounding_box_coords.split(";")

        #create arrays of boxes

        boxes=list()
        for i in boxes_string:
            coords=i.split()
            boxes.append(coords)
        boxes_np=np.array(boxes)
        boxes_np=boxes_np.astype(np.float32)

        boxes_tensor=torch.tensor(boxes_np)


        return image_tensor_normalized,boxes_tensor
    
    def collate_fn(self,batch):
        images=list()
        boxes=list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
        
        images=torch.stack(images,dim=0)

        return images,boxes