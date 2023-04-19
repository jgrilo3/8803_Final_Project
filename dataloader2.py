import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy
import matplotlib.pyplot as plt
import csv


LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}


mean = (.1706)
std = (.2112)
normalize = transforms.Normalize(mean=mean, std=std)

transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    normalize,
])
    
class OCTDataset(Dataset):
    def __init__(self, root, subset='train', transform=None,):
        if subset == 'train':
            self.annot = pd.read_csv(root + '/df_prime_train.csv')
        elif subset == 'test':
            self.annot = pd.read_csv(root + '/df_prime_test.csv')
        self.subset = subset    
        self.annot['Severity_Label'] = [LABELS_Severity[drss] for drss in copy.deepcopy(self.annot['DRSS'].values)] 
        # print(self.annot)
        self.root = os.path.expanduser(root)
        self.transform = transform
        # self.subset = subset
        self.nb_classes=len(np.unique(list(LABELS_Severity.values())))
        #self.path_list = self.annot['File_Path'].values
        temp_path_list = self.annot['File_Path'].values
        path_indices = [i for i, _ in enumerate(temp_path_list) if not np.isnan(self.annot["BMI"].values[i])]
        self.path_list = [self.annot['File_Path'].values[i] for i in path_indices]

        self._labels = [self.annot['Severity_Label'].values[i] for i in path_indices]
        assert len(self.path_list) == len(self._labels)
        # idx_each_class = [[] for i in range(self.nb_classes)]

    def __getitem__(self, index,meta=False):
        if self.subset == 'train':
            csv_file = open(self.root + '/df_prime_train.csv', 'r')
        if self.subset == 'test':
            csv_file = open(self.root + '/df_prime_test.csv', 'r')
        csv_reader = csv.reader(csv_file)
         
        for j, row in enumerate(csv_reader):
            if self.path_list[index] == row[2]:
                metadata = row[8:17]
                if np.isnan(float(metadata[5])):
                    print(metadata)
                metadata = [float(x) for x in metadata]
                metadata = torch.tensor(metadata)
                break
        csv_file.close()

        img, target = Image.open(self.root + self.path_list[index]).convert("L"), self._labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        #print(self.path_list[index])
        #print(metadata)

        return img, target, metadata

    def __len__(self):
        return len(self._labels)         


if __name__ == '__main__':
    root =  'C:/Users/jgril/Documents/GitHub/8803_Final_Project'
    trainset = OCTDataset(root, 'train', transform=transform)
    testset = OCTDataset(root, 'test', transform=transform)
    print("finished")
    #image = plt.imshow(np.squeeze(trainset.__getitem__(0)[0], axis=0))
    #plt.show()
    print(trainset.__getitem__(0)[0].flatten())
    #print(trainset[0][0].shape)
    #print(len(trainset), len(testset))
    #print(trainset.path_list)
    #print(trainset._labels)