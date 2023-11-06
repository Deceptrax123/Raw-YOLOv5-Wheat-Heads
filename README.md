# YOLOv5

This repository is a part of Negative-Transfer-Mitigation. It contains various
config and helper files required to train a custom YOLOv5 model using Ultralytics and test
the results on AI Crowd. 

# Description

## Data_Yamls Folder
YAML file containing dataset paths and class names

## Utils

| Name | Description|
|-----|------------|
|create_data_folders.py| Code to perform train-test-val split for YOLOv5 and move the images to the respective folders|
|generate_annotations.py| Script to convert annotations to YOLOv5 Format|
|infer.py|Script to convert annotations from YOLOv5 format to PASCAlVOC format|


## Metrics
Contains the results for Raw YOLOv5 trained on the Wheat Head Dataset

## Merge Notebook 
Domain Mapping of Test Set of Wheat Head Dataset using the sample solutions file