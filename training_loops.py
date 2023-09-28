import torch 
from torch import nn
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
import numpy as np
import torchvision.ops as O


def train_epoch():
    epoch_loss=0

    for step,(sample,y_box) in enumerate(train_loader):
        sample=sample.to(device=device)
        box=box.to(device=device)

        #get bounding box predictions
        predictions_box=model(sample)
        
        #train model
        model.zero_grad()
        box_losses=bbox_loss(predictions_box,y_box)
        box_losses.backward()
        optimizer.step()

        epoch_loss+=box_losses.item()

        #Memory Management
        del sample
        del y_box
        del predictions_box
        del box_losses
        mps.empty_cache()
    
    mean_loss=epoch_loss/train_steps

    return mean_loss

def test_epoch():
    loss=0

    for step,(sample,y_box) in enumerate(test_loader):
        sample=sample.to(device=device)
        y_box=y_box.to(device=device)

        #evaluate test split
        predictions_box=model(y_box)

        test_loss=O.generalized_box_iou(predictions_box,y_box)

        test_loss=torch.mean(test_loss)
        loss+=test_loss.item()

        #Memory Management
        del predictions_box
        del test_loss
        del sample
        del y_box

        mps.empty_cache()
    
    mean_loss=loss/test_steps

    return mean_loss

def training_loop():
    for epoch in range(NUM_EPOCHS):
        model.train(True)
        train_loss=train_epoch()
        
        model.eval()

        with torch.no_grad():
            test_loss=test_epoch()
            print("Epoch {epoch}".format(epoch=epoch+1))
            print("Train Bounding Box Loss : {tloss}".format(tloss=train_loss))
            print("Test Bounding Box IOU {test_loss}".format(test_loss=test_loss))

            wandb.log({
                "Train Box Loss":train_loss,
                "Test Box IOU":test_loss,
                }
            )

        #checkpoints
        if (epoch+1)%10==0:
            path="./trained_weights/yolov5/run_1/model{epoch}.pth".format(epoch=epoch+1)
            torch.save(model.state_dict(),path)



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

    wandb.init(
        project="wheat-heads-detection",
        config={
            "architecture":"YOLOv5",
            "dataset":"Wheat Heads Dataset"
        },
    )

    if torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")

    #Hyperparameters
    LR=0.01
    BETAS=(0.9,0.999)
    NUM_EPOCHS=20

    #load yolov5
    model=torch.hub.load("ultralytics/yolov5",'yolov5s',autoshape=False,pretrained=True).to(device=device)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR,betas=BETAS)


    #Set Train and test steps
    train_steps=(len(train)+params_train['batch_size']-1)//params_train['batch_size']
    test_steps=(len(test)+params_test['batch_size']-1)//params_test['batch_size']

    #Loss functions
    bbox_loss=nn.SmoothL1Loss()

    training_loop()