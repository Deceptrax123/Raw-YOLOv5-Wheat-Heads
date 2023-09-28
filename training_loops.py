import torch 
import torchvision.transforms.v2 as T 
from torch.utils.data import DataLoader
from wheat_dataset import WheatDataset
import torch.multiprocessing
import wandb
from time import time 
from torch import mps,cpu 
import gc 
from sklearn.model_selection import train_test_split
import pandas as pd 


if __name__=='__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    dataset=pd.read_csv("./train/train.csv")

    image_names=list(dataset['image_name'])
    bounding_boxes=list(dataset['BoxesString'])

    data=dict(zip(image_names,bounding_boxes))

    train,test=train_test_split(image_names,test_size=0.25)


    train_set=WheatDataset(image_names=train,data_dict=data)
    test_set=WheatDataset(image_names=test,data_dict=data)

    params_train={
        'batch_size':8,
        'shuffle':True,
        'num_workers':0,
        'collate_fn':train_set.collate_fn
    }

    params_test={
        'batch_size':8,
        'shuffle':True,
        'num_workers':0,
        'collate_fn':test_set.collate_fn
    }

    train_loader=DataLoader(train_set,**params_train)
    test_loader=DataLoader(test_set,**params_test)


    #load yolov5
    model=torch.hub.load("ultralytics/yolov5",'yolov5s',autoshape=False,pretrained=False)

    