import os
import time 
import numpy as np
from glob import glob
import re
from PIL import Image
import cv2 as cv2
from tqdm import tqdm,tnrange
from itertools import chain
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

def gray2rgb(img_gray):
    # site maps are in gray-scale, replacte the same image twcie for the other two channels 
    # since most pre-trained models taks RGB data
    # apply after padding
    img_rgb = img_gray.repeat_interleave(3,0)
    return img_rgb


# pad images in the dataset  on both side
def pad4sides_func(img,desired_dim,fill):
    left = int((desired_dim[1] - img.shape[2])/2)
    right = desired_dim[1]- img.shape[2]-left
    top = int((desired_dim[0] - img.shape[1])/2)
    bottom = desired_dim[0] - img.shape[1]-top
    padded = torch.nn.functional.pad(img,(left,right,top,bottom),value= fill)
    #img = pad(img)
    return padded

class pad4sides(object):
    def __init__(self, desired_dim,fill=0.0):
       
        self.desired_dim = desired_dim
        self.height  = desired_dim[0]
        self.width  = desired_dim[1]
        self.fill = fill
        
    def __call__(self, sample):
        """
        Args:
            sample = dtm/hillshade/mask

        Returns: Padded image and mask
        """

        sample_padded = pad4sides_func(sample,self.desired_dim,self.fill)
        return sample_padded
        
    def __repr__(self):
        return self.__class__.__name__ + f'(fill={self.fill}, padding_mode={self.mode})'


def get_all_transformed(dataset):
    assert isinstance(dataset,torch.utils.data.dataset.Subset) or issubclass(type(dataset), Dataset)
    loader = DataLoader(dataset, batch_size=len(dataset))
    for X,y in loader:
        X_T = X
        y_T = y
    return X_T,y_T           

# pad on right and bottom only 
def pad_ri_bot_func(img,desired_dim,fill):
    #left = int((desired_dim[1] - img.shape[2])/2)
    right = desired_dim[1]- img.shape[2]
    #top = int((desired_dim[0] - img.shape[1])/2)
    bottom = desired_dim[0] - img.shape[1]
    padded = torch.nn.functional.pad(img,(0,right,0,bottom),value= fill)
    #img = pad(img)
    return padded

class pad_right_bottom(object):
    def __init__(self, desired_dim,fill=0.0):      
        self.desired_dim = desired_dim
        self.height  = desired_dim[0]
        self.width  = desired_dim[1]
        self.fill = fill        
    def __call__(self, sample):
        sample_padded = pad_ri_bot_func(sample,self.desired_dim,self.fill)
        return sample_padded        
    def __repr__(self):
        return self.__class__.__name__ + f'(fill={self.fill}, padding_mode={self.mode})'

    


def split_data(fullDataset,train_prop,val_prop=0.0,seed=42):
    # splilt data into train,val,test, 
    # return train_set, val_set, test_set
    # if not given a val_prop, returned train and test 
    make_val = val_prop!= 0.0
    train_size = int(np.rint((train_prop+val_prop)*len(fullDataset)))
    test_size = len(fullDataset)-train_size
    trainDataset,testDataset = torch.utils.data.random_split(
        fullDataset,[train_size,test_size], 
        generator=torch.Generator().manual_seed(seed))
    if not make_val:
        print(f"train:test = {len(trainDataset)}:{len(testDataset)}")
        return trainDataset,testDataset
    else: # split train again into train and val
        val_size = int(np.rint(val_prop*len(fullDataset)))
        train_size = len(trainDataset)-val_size
        trainDataset,valDataset = torch.utils.data.random_split(
            trainDataset,[train_size,val_size], 
            generator=torch.Generator().manual_seed(seed))
        print(f"train:val:test = {len(trainDataset)}:{len(valDataset)}:{len(testDataset)}")
        return trainDataset,valDataset,testDataset



class HELMDataset(Dataset):
    # data stucture for prediction using pytorch dataset structure
    def __init__(self, dtm_og=None, hsd_og=None, label_og=None, 
                subdtm_arrays=None, subhsd_arrays=None, sublabel_arrays=None,coord_list=None, 
                use_hillshade=True, drop_edge = True, size = 512,transform=None,pad_transform= None):
        # input: hillshade and label images as numpy array, dtm is optional
        self.completehsd = hsd_og
        self.completelabel = label_og
        self.completedtm = dtm_og
        self.subdtms = []
        self.sublabels = []
        self.subhsds = []
        self.coords = []
        self.transform = transform
        self.pad_transform = pad_transform
        self.usehillshade = use_hillshade

        # match corresponding label png given an image
        for i in tqdm(range(len(subhsd_arrays)),desc="adding hillshade and label"):
            # Read an image with PIL, convert to np array
            sub_hsd = subhsd_arrays[i]
            sub_label = sublabel_arrays[i]
            if dtm_og != None and subdtm_arrays !=None:
                sub_dtm = subdtm_arrays[i]
            x,y = coord_list[i]
            if drop_edge:
                if np.shape(sub_hsd) == (size,size):
                    self.sublabels.append(sub_label)
                    self.subhsds.append(sub_hsd)
                    self.coords.append(coord_list[i])
                    if dtm_og != None and subdtm_arrays !=None:
                        self.subdtms.append(sub_dtm)
                else:
                    next
            else:
                self.sublabels.append(sub_label)
                self.subhsds.append(sub_hsd)
                self.coords.append(coord_list[i])
                if dtm_og != None and subdtm_arrays !=None:
                    self.subdtms.append(sub_dtm)
        
    def __len__(self):
        return len(self.sublabels)


    def __getitem__(self, idx):
        if self.usehillshade:
            hsd = self.subhsds[idx]
            label = self.sublabels[idx].astype("int32")
            # apply the transformations to both image and its label
            if self.pad_transform is not None:
                # pad image/label to subimages of the same size, convert to tensor
                # dimension = [h w c] -> [c h w],
                hsd_T = self.pad_transform(hsd.copy())
                label_T = self.pad_transform(label.copy())
            else:
                label_T,hsd_T = label.copy(),hsd.copy()
            # check to see if we are applying any transformations
            if self.transform is not None:
                hsd_T = self.transform(hsd_T)
            else:
                hsd_T  = hsd_T
            return hsd_T,label_T
        else: # if not hillshade assume using DTM
            dtm = self.subdtms[idx]
            label = self.sublabels[idx].astype("int32")
            # apply the transformations to both image and its label
            if self.pad_transform is not None:
                # pad image/label to subimages of the same size, convert to tensor
                # dimension = [h w c] -> [c h w],
                dtm_T = self.pad_transform(dtm.copy())
                label_T = self.pad_transform(label.copy())
            else:
                label_T,dtm_T = label.copy(),dtm.copy()
            # check to see if we are applying any transformations
            if self.transform is not None:
                dtm_T = self.transform(dtm_T)
            else:
                dtm_T  = dtm_T
            return dtm_T,label_T

    def getAll3(self, idx):
        
        dtm = self.subdtms[idx]
        label = self.sublabels[idx]
        hsd = self.subhsds[idx]

        # apply the transformations to both image and its label
        if self.pad_transform is not None:
            # pad image/label to subimages of the same size, convert to tensor
            # dimension = [h w c] -> [c h w],
            dtm_T = self.pad_transform(dtm)
            label_T = self.pad_transform(label)
            hsd_T = self.pad_transform(hsd)
        else:
            label_T = label

        # check to see if we are applying any transformations
        if self.transform is not None:
            dtm_T = self.transform(dtm_T)
            hsd_T = self.transform(hsd_T)
        else:
            dtm_T  = dtm
            hsd_T = hsd 

        return dtm_T,hsd_T,label_T
        
    def count_labels(self):

        unique_per_label =[np.unique(self.sublabels[i]) for i in range(len(self))]
        unique_classes = list(set(chain(*unique_per_label)))
        self.num_classes = len(unique_classes)
        return len(unique_classes)
