import torch 
import pandas as pd 
import numpy as np 
from PIL import Image
from sklearn.model_selection import train_test_split
import os 

#global variables

mapping={
    "wheat_head":1,
    "background":0
}

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

def prepare_for_yolov5(data_details):

    file_list=list()
    for box in data_details['bboxes']:

        class_id=mapping[box['class']]
        #Transform to yolo v5 format
        box_center_x=(box['xmin'])+(box['xmax'])/2
        box_center_y=(box['ymin']+box['ymax'])/2
        box_width=box['xmax']-box['xmin']
        box_height=box['ymax']-box['ymin']

        h,w,c=data_details['image_size']

        box_center_x=box_center_x/h
        box_center_y=box_center_y/w
        box_width=box_width/h
        box_height=box_height/w

        file_list.append("{} {:.3f} {:.3f} {:.3f}".format(class_id,box_center_x,box_center_y,box_width,box_height))
    
    #save the coordinates to a text file as required by Yolo v5
    file_name="./annotations/"+data_details['filename'].replace("png","txt")
    
    #write annotations to file
    with open(file_name,'w') as f:
        for i,item in enumerate(file_list):
            f.write(item)
            
            if i!=len(file_list)-1:
                f.write("\n")
        f.close()