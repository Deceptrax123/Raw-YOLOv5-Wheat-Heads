import numpy as np 
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import shutil

def files_to_folder(files,destination):
    for f in files:
        shutil.move(f,destination)



images=[os.path.join('./train/train/',x) for x in os.listdir('./train/train/')]
annotations=[os.path.join('./annotations/',x) for x in os.listdir('./annotations/')]

images.sort()
annotations.sort()

#split dataset as train, test and val
train_images,val_images,train_annots,val_annots=train_test_split(images,annotations,test_size=0.25,random_state=1)
val_images,test_images,val_annots,test_annots=train_test_split(val_images,val_annots,test_size=0.5,random_state=1)

#move images to their respective folders
files_to_folder(train_images,"./images/train/")
files_to_folder(val_images,"./images/val/")
files_to_folder(test_images,"./images/test/")
files_to_folder(train_annots,"./labels/train/")
files_to_folder(val_annots,"./labels/val/")
files_to_folder(test_annots,"./labels/test/")