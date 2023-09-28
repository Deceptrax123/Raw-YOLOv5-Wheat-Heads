import torch 
import pandas as pd 
import numpy as np 
from PIL import Image
from sklearn.model_selection import train_test_split


def create_dict():
    data=pd.read_csv("./train/train.csv")

    image_names=data['image_name']
    box_coords=data['BoxesString']

    data_dict=dict(zip(image_names,box_coords))

    global_path="./train/train/"
    
    data_details=list()
    for i,key in enumerate(data_dict):
        coordinates=data_dict[key]

        if coordinates!='no_box':
            boxes_string=coordinates.split(";")

            boxes=list()
            for i in boxes_string:
                coords=i.split()
                xmin,ymin,xmax,ymax=int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3])

                box_dict={
                    'class':'wheat_head',
                    'xmin':xmin,
                    'ymin':ymin,
                    'xmax':xmax,
                    'ymax':ymax
                }
                boxes.append(box_dict)
        else:
            boxes=[{
                'class':'background',
                'xmin':0,
                'ymin':0,
                'xmax':0,
                'ymax':0
            }]

        img=Image.open(global_path+key+".png")
        img_np=np.array(img)
        image_dict={
            'bboxes':boxes,
            'filename':key+".png",
            'image_size':img_np.shape,
        }

        data_details.append(image_dict)

        if i==0:
            break

    return data_details