#prepare dataframe 

import pandas as pd 
import numpy as np 
import os 

def create_df():
    paths=os.listdir("YOLOv5_config/labels/")

    img_names=list()
    boxes=list()
    for pa in paths:
        file_path="YOLOv5_config/labels/"+pa
        with open(file_path,'r') as f:
            items=f.readlines()
            box=''
            for k,i in enumerate(items):             
                sp=i.split()
                x_mid,y_mid,x_w,y_w=float(sp[1]),float(sp[2]),float(sp[3]),float(sp[4].replace('\n',''))

                #Convert normalized form to actual form
                x_mid=x_mid*1024
                y_mid=y_mid*1024
                x_w=x_w*1024
                y_w=y_w*1024

                #get format xmin ymin xmax ymax
                x_max=(x_w/2)+x_mid
                y_max=(y_w/2)+y_mid
                x_min=x_max-x_w
                y_min=y_max-y_w

                #round 
                x_max=round(x_max)
                y_max=round(y_max)
                x_min=round(x_min)
                y_min=round(y_min)

                if k!=len(items)-1: #check last item
                    sequence=str(x_min)+' '+str(y_min)+' '+str(x_max)+' '+str(y_max)+';'
                else:
                    sequence=str(x_min)+' '+str(y_min)+' '+str(x_max)+' '+str(y_max)
                
                box=box+sequence
            f.close()
        img_names.append(pa.replace('.txt',''))
        boxes.append(box)

    #crete pandas datadrame
    print(len(img_names))
    df=pd.DataFrame({'image_name':img_names,'BoxString':boxes})

    df.to_csv("predictions.csv")

create_df()