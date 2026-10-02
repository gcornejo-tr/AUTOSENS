import json
import os
import os.path as op
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyDataset(Dataset):
    def __init__(self, folder_path, sequences = None):

        self.folder_path = folder_path
        self.sequences = sequences
        self.imgdir, self.args_dict = self.get_imgdir_labels(sequences)

    def __len__(self):
        return len(self.imgdir)

    def __getitem__(self, idx):

        image = Image.open(self.imgdir[idx]).convert('RGB')
        transform = transforms.Compose([transforms.ToTensor(),])

        image_tensor = transform(image)
        
        if self.args_dict['x_min'][idx]:

            class_pedestrian = 0.0
            annotation_tensor = torch.tensor(
                [self.args_dict['x_min'][idx], self.args_dict['y_min'][idx], 
                self.args_dict['x_max'][idx], self.args_dict['y_max'][idx], 
                class_pedestrian, self.args_dict['dist'][idx]])
            
            distance = torch.tensor(self.args_dict['dist'][idx])

        else:

            annotation_tensor = torch.tensor([float('nan'), float('nan'), float('nan'), float('nan'), -1, float('nan')])
            distance = torch.tensor(float('nan'))

        imagedir = self.args_dict['folder'][idx] + self.args_dict['frame_name'][idx]

        dictData = {"idx": idx, 
                    "image_tensor": image_tensor,
                    "annotation_tensor": annotation_tensor,
                    "distance": distance, 
                    "imagedir": imagedir}

        return dictData
    
    def get_imgdir_labels(self, sequences):

        directions = []

        if op.exists(self.folder_path) and op.isdir(self.folder_path):

            for root, _ , files in os.walk(self.folder_path):
                if root in sequences:
                    for file in files:
                        directions.append(op.join(root, file))

        imgdir = []
        args_dict = {'x_min':[],'y_min':[], 'x_max':[], 'y_max':[], 'dist': [], 
                     'frame_name': [], 'folder': []}

        directions.sort()

        for dir in directions:
            if dir[-3:]=="png":
                imgdir.append(dir)
                
            if dir[-4:]=="json":
                with open(dir, 'r') as f:
                    data = json.load(f)
                    args_dict['frame_name']=args_dict['frame_name']+data['frame_name']
                    args_dict['folder']=args_dict['folder']+ [data['folder']] * len(data['frame_name'])

                    if data['folder'][-1] == 'p':

                        args_dict['x_min']=args_dict['x_min']+[o['location_2d'][0] for o in data['object']]
                        args_dict['y_min']=args_dict['y_min']+[o['location_2d'][1] for o in data['object']]
                        args_dict['x_max']=args_dict['x_max']+[o['location_2d'][2] for o in data['object']]
                        args_dict['y_max']=args_dict['y_max']+[o['location_2d'][3] for o in data['object']]
                        args_dict['dist']=args_dict['dist']+[o['location_3d'][0] for o in data['object']] #dist        

                    else:
                        args_dict['x_max']=args_dict['x_max']+[None] * len(data['frame_name'])
                        args_dict['x_min']=args_dict['x_min']+[None] * len(data['frame_name'])
                        args_dict['y_max']=args_dict['y_max']+[None] * len(data['frame_name'])
                        args_dict['y_min']=args_dict['y_min']+[None] * len(data['frame_name'])
                        args_dict['dist']=args_dict['dist']+[None] * len(data['frame_name'])

        for image1, image2 in zip(imgdir,args_dict['frame_name']):
            assert image1[-10:]==image2[-10:], "Data not loaded correctly"  

        return imgdir, args_dict
        






