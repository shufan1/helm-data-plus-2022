import os
import re
import argparse
import sys
import numpy as np
import os
from glob import glob
import re
from tqdm import tqdm
from PIL import Image
import cv2 as cv2
from torch.utils.data import Dataset

import argparse
import matplotlib.ticker as ticker
from itertools import chain

"""
helper functions for preprocessing: 
    including: split images, set up datastructure for applying transformation, flip and rotate transformation
used mainly by 01_preparedata.py
"""

siteRegex = re.compile(
    "Kilbixy|Fore|Glenogra|Ardmayle|Kilmacahill|Ballynahnich|Knockainey|Knockainy", 
    re.IGNORECASE) # used to find corresponding label file given a hillshadr or dtm file


def split_w_overlap(hillshade_path,label_path=None,dtm_path = None, siteName=None,
            chipsize=512,overlap=0,write=False,output_site_dir=None):
    """
    make folder under output_root given a site, 
    split the big site hillshade and labels into suquare subimages imgs and labels and dtm will be saved separately
    label and dtm are optional
    input: full path to hillshade file, label file and the dtm file
    chipsize = sub image size, 
    optional to write out the subimage. If yes, write as .npy file
    """       
 
    split_dtm  = dtm_path !=  None
    split_label = label_path != None
    
    if write: # if need to write output, create output folder organized by site > hillshade, labels and dtms
        output_hillshade_dir = os.path.join(output_site_dir,"hillshades")
        os.makedirs(output_hillshade_dir,exist_ok= True)
        os.makedirs(output_site_dir,exist_ok= True)
        if split_label:
            output_label_dir = os.path.join(output_site_dir,"labels")
            os.makedirs(output_label_dir,exist_ok= True)
        if split_dtm:
            output_dtm_dir = os.path.join(output_site_dir,"dtms")
            os.makedirs(output_dtm_dir,exist_ok= True)

    hillshade = cv2.imread(hillshade_path,flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
    if split_label:
        label = cv2.imread(label_path,flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
        label[label==2]=1
    if split_dtm:
        dtm = cv2.imread(dtm_path,flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))

    coord_list = [] # store the coordiante of the upper left corner

    img = hillshade
    total_sub = len(range(0,img.shape[0],chipsize-overlap))*len( range(0,img.shape[1],chipsize-overlap))
    sub_labels = [[] for i in range(total_sub)]
    sub_hsds = [[] for i in range(total_sub)]
    sub_dtms = [[] for i in range(total_sub)]
    coord_list = [[] for i in range(total_sub)]
    i = 0
    for r_start in tqdm(range(0,img.shape[0],chipsize-overlap),desc=f"spliting {siteName} into {total_sub} subimages"):
        for c_start in range(0,img.shape[1],chipsize-overlap):
            # handling spiting at the edge of the image
            if r_start > img.shape[0]-chipsize:
                r_end = img.shape[0]
            else:
                r_end = r_start+chipsize
            if c_start > img.shape[1]-chipsize:
                c_end = img.shape[1]
            else:
                c_end = c_start+chipsize
            coord_list[i] = (r_start,c_start)

            if write:
                subhillshade_path = os.path.join(output_hillshade_dir,f"{siteName}{r_start}_{c_start}.png")
                subhsd = Image.fromarray(hillshade[r_start:r_end, c_start:c_end]).convert("L")
                subhsd.save(subhillshade_path)  
            sub_hsds[i] = hillshade[r_start:r_end, c_start:c_end]
            if split_label:
                if write:
                    sublabel_path = os.path.join(output_label_dir,f"{siteName}{r_start}_{c_start}.png")
                    sublabel = Image.fromarray(label[r_start:r_end, c_start:c_end]).convert("L")
                    sublabel.save(sublabel_path)
                sub_labels[i] = label[r_start:r_end, c_start:c_end] 
            if split_dtm:
                if write:
                    subdtm_path = os.path.join(output_dtm_dir,f"{siteName}{r_start}_{c_start}.png")
                    subdtm = Image.fromarray(dtm[r_start:r_end, c_start:c_end]).convert("L")
                    subdtm.save(subdtm_path)
                sub_dtms[i] = dtm[r_start:r_end, c_start:c_end] 

            # if write:
            #     subhillshade_path = os.path.join(output_hillshade_dir,f"{siteName}{r_start}_{c_start}")
            #     subhsd = hillshade[r_start:r_end, c_start:c_end]
            #     np.save(subhillshade_path,hillshade[r_start:r_end, c_start:c_end])  
            # sub_hsds[i] = hillshade[r_start:r_end, c_start:c_end]
            # if split_label:
            #     if write:
            #         sublabel_path = os.path.join(output_label_dir,f"{siteName}{r_start}_{c_start}")
            #         np.save(sublabel_path,label[r_start:r_end, c_start:c_end])
            #     sub_labels[i] = label[r_start:r_end, c_start:c_end] 
            # if split_dtm:
            #     if write:
            #         subdtm_path = os.path.join(output_dtm_dir,f"{siteName}{r_start}_{c_start}")
            #         np.save(subdtm_path,dtm[r_start:r_end, c_start:c_end])
            #     sub_dtms[i] = dtm[r_start:r_end, c_start:c_end] 

            i += 1

    return sub_dtms,sub_hsds,sub_labels,coord_list


class augmentDataset(Dataset):
    # data strcture for applying transfomration easier
    # use index to get sample: image and label at the same time
    def __init__(self, dtm_list, hsd_list,label_list, coord_list, transform=None,transform_name = None):       
        self.dtms = []
        self.hillshades = []
        self.labels = []
        self.dtmstransformed = []
        self.hillshadestransformed = []
        self.labelstransformed = []
        self.transform = transform
        self.transform_name = transform_name
        self.imgName_list = []
        # match corresponding label png given an image
        for i in tqdm(range(len(dtm_list)),desc="adding original sub hillshade images, its corresponding label and dtm"):
            self.dtms.append(dtm_list[i])
            self.labels.append(label_list[i])
            self.hillshades.append(hsd_list[i])
            self.imgName_list.append(str(coord_list[i][0])+"_"+str(coord_list[i][1]))
   
    def __len__(self):
        return len(self.dtms)

    def __getitem__(self, idx):
        dtm = self.dtms[idx]
        label = self.labels[idx]
        hillshade = self.hillshades[idx]
        # apply the transformations to both image and its label
        if self.transform is not None:
            dtm_T = self.transform(dtm)
            label_T = self.transform(label)
            hillshade_T = self.transform(hillshade)
        else:
            dtm_T = dtm
            label_T = label
            hillshade_T =hillshade
        return dtm_T, hillshade_T,label_T

    def get_all_transformed(self,siteName = None,writeImg=False,output_dir=None,):
        # apply transformation to all the sub hillshde and labels, as well as dtm dataset
        # write transformed numpy array as image in .png format is needed
        if writeImg:
            os.makedirs(os.path.join(output_dir,"dtms"),exist_ok=True)
            os.makedirs(os.path.join(output_dir,"hillshades"),exist_ok=True)
            os.makedirs(os.path.join(output_dir,"labels"),exist_ok=True)

        self.dtmstransformed = ["" for i in range(len(self))]
        self.labelstransformed = ["" for i in range(len(self))]
        self.hillshadestransformed = ["" for i in range(len(self))]
        for i in tqdm(range(len(self)),desc=f"applying {self.transform_name}, saving transformed images and labels"):
            imgName = self.imgName_list[i]
            if self.transform != None:
                # apply transformation to both the image and the label
                dtm_T= self.transform(self.dtms[i])
                label_T= self.transform(self.labels[i]) 
                hillshade_T = self.transform(self.hillshades[i]) 
                # update trasnsformed images and labels list
                self.dtmstransformed[i]=dtm_T
                self.labelstransformed[i]=label_T
                self.hillshadestransformed[i] = hillshade_T
            else: #simplying copy:
                dtm_T,hillshade_T, label_T =self[i]

            if writeImg:
                dtm_T_file = Image.fromarray(dtm_T).convert("L")
                dtmfilename = os.path.join(output_dir ,f"dtms/{siteName}{imgName}{self.transform_name}.png")
                dtm_T_file.save(dtmfilename)
                label_T_file = Image.fromarray(label_T).convert("L")
                labelfilename = os.path.join(output_dir ,f"labels/{siteName}{imgName}{self.transform_name}.png")
                label_T_file.save(labelfilename)
                hillshade_T_file = Image.fromarray(hillshade_T).convert("L")
                hillshadefilename = os.path.join(output_dir ,f"hillshades/{siteName}{imgName}{self.transform_name}.png")
                hillshade_T_file.save(hillshadefilename)

            # if writeImg:
            #     dtmfilename = os.path.join(output_dir ,f"dtms/{imgName}{self.transform_name}")
            #     np.save(dtmfilename,dtm_T)
            #     labelfilename = os.path.join(output_dir ,f"labels/{imgName}{self.transform_name}")
            #     np.save(labelfilename,label_T)
            #     hsdfilename = os.path.join(output_dir ,f"hillshades/{imgName}{self.transform_name}")
            #     np.save(hsdfilename,hillshade_T)

        return self.dtmstransformed , self.hillshadestransformed, self.labelstransformed


# flip transformation
class flip(object):
    def __init__(self,flip_dir):
        self.flip_dir = flip_dir
        self.flip_dim = 1 if flip_dir == "horizontal" else 0

    def __call__(self, sample):
        """
        Args:
            sample = img/label, one of them
        Returns: Flipped img or label
        """
        sample_flipped = np.flip(sample,self.flip_dim)
        return sample_flipped
        
    def __repr__(self):
        return self.__class__.__name__ + f'(flip {self.flip_dir}ly along dim {self.flip_dim})'

# rotate 90, 180, 270 CCW transformation
class rotate(object):
    def __init__(self,degree):
        self.degree = degree
        self.rotate_times = self.degree//90
    def __call__(self, sample):
        """
        Args:
            sample = img/label one of them
        Returns: Flipped img or label
        """
        sample_rotated = np.rot90(sample,self.rotate_times, (0,1))
        return sample_rotated
        
    def __repr__(self):
                return self.__class__.__name__ + f'(rotate 90 degree {self.rotate_times} times CCW)'




