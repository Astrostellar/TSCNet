import torch
import numpy as np
import argparse
from train_stage_1 import train_stage_1
from train_stage_2 import train_stage_2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--data_path", type=str, default="../../data/public_data/SKI10/test", help="path to the dataset")
    parser.add_argument("--batch-size", type=int, default=6, help="batch size")
    parser.add_argument("--sample-size", type=int, default=256, help="the size of the patches")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--thick_direction", type=str, default="axial", help="the slices within the plane is the thick slice, select among ['axial', 'sagittal', 'coronal']")
    parser.add_argument("--project_name", type=str, default='SKI10', help="the name of the project")
    parser.add_argument("--stage", type=int, default=1, help="the stage for training, choose from [1, 2, 3]")
    parser.add_argument("--resume", type=str, default=None, help="the path of the model to resume")
    
    args = parser.parse_args()
    if args.stage==1:
        train_stage_1(args)
    elif args.stage==2:
        train_stage_2(args)
    elif args.stage==3:
        train_stage_1(args)
        train_stage_2(args)




    
