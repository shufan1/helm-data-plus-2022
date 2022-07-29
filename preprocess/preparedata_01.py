# split images and labels in 00_Data_Preprocessed with overlaps.
# same way to split each image and its corresponding label, flip rotate to get more replicates
# save results by site in 01_Data_Prepared
# split with overlap: https://github.com/Devyanshu/image-split-with-overlap/blob/master/split_image_with_overlap.py

from ast import arg
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
from matplotlib import pyplot as plt
import argparse
import matplotlib.ticker as ticker
from itertools import chain
from utils.preprocess_utils import split_w_overlap, augmentDataset, flip, rotate

#plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


siteRegex = re.compile(
    "Kilbixy|Fore|Glenogra|Ardmayle|Kilmacahill|Ballynahnich|Knockainey|Knockainy", 
    re.IGNORECASE) # used to find corresponding label file given a hillshadr or dtm file



def split_transform(hsd_path,label_path,dtm_path,siteName,
                    chipsize,overlap,transform_list=None,transform_name_list=None,
                    write=False,output_root = None ):

    """
    given the hillshade, dtm and label of a site, apply split with overlap and a list of transformations 
    this step is augmentation. Reduce sample dimension for Unet and create more versions of the origninal data 
    Writing out the output is optional
    output is in the format of a dictionary following this sturcture: 
    { "og": {"dtm": list of 2D array, "hsd": list of 2D array, "label":list of 2D array},
        "transformation1_name": {"dtm": list of 2D array, "hsd": list of 2D array, "label":list of 2D array},
        ...}
    And the coordinate of the upper left corner 

    """
    # split with overlap
    subdtms_og,subhsds_og,sublabels_og,coord_list = split_w_overlap(hsd_path,label_path,dtm_path,siteName = siteName,
                                                                        chipsize=chipsize,overlap=overlap,
                                                                        write=write,output_site_dir=output_root)#,write=True,output_root=output_root)
    
    transformed_result = {'og':{"dtm":0,"hsd":0,"label":0,"subcoords":0}}
    transformed_result['og']['dtm'] = subdtms_og
    transformed_result['og']['hsd'] = subhsds_og
    transformed_result['og']['label'] = sublabels_og
    transformed_result['og']['subcoords'] = coord_list
    # apply a list of transformation save the results in a dictionary
    if transform_list != None:
        for s in transform_name_list:
            transformed_result[s] = {"dtm":0,"hsd":0,"label":0} 
        for i in range(len(transform_list)):
            HELM_transformed = augmentDataset(subdtms_og,subhsds_og,sublabels_og,coord_list, transform_list[i],transform_name_list[i])
            # apply transformation and save images if needed
            subdtms_T , subhsds_T, sublabels_T= HELM_transformed.get_all_transformed(siteName,writeImg=write, output_dir=output_root)
            transformed_result[transform_name_list[i]]['dtm'] = subdtms_T
            transformed_result[transform_name_list[i]]['hsd'] = subhsds_T
            transformed_result[transform_name_list[i]]['label'] = sublabels_T
            transformed_result[transform_name_list[i]]['subcoords'] = coord_list

    return transformed_result


def unpack_transformed_result(transformed_result):
    # unpkack all transformed sub-dtm, sub-hsd and sub-label into a list
    subdtm_augmented_all = [transformed_result[s]['dtm'] for s in transformed_result]
    subdtm_augmented_all  = list(chain(*subdtm_augmented_all ))
    subhsd_augmented_all = [transformed_result[s]['hsd'] for s in transformed_result]
    subhsd_augmented_all  = list(chain(*subhsd_augmented_all ))
    sublabel_augmented_all = [transformed_result[s]['label'] for s in transformed_result]
    sublabel_augmented_all  = list(chain(*sublabel_augmented_all ))
    subcoords_augmented_all = [transformed_result[s]['subcoords'] for s in transformed_result]
    subcoords_augmented_all  = list(chain(*subcoords_augmented_all ))

    return subdtm_augmented_all,subhsd_augmented_all,sublabel_augmented_all,subcoords_augmented_all


def visualize_transformations(transformed_result,example_idx,title):
    """
    visualize transformation given the dictionary of the transformation, and example idx in the original splitted subimages.
    """
    i = example_idx
    dtm_vis_list = [transformed_result[s]['dtm'][i] for s in transformed_result]
    hsd_vis_list = [transformed_result[s]['hsd'][i] for s in transformed_result]
    label_vis_list = [transformed_result[s]['label'][i] for s in transformed_result]

   
    fig,ax = plt.subplots(len(dtm_vis_list),3,figsize=(12,len(dtm_vis_list)*12/3))
    for i in range(len(dtm_vis_list)):
        ax[i][0].imshow(dtm_vis_list[i],cmap="gray")
        ax[i][0].set_ylabel(title[i],fontsize=30)
        ax[i][0].xaxis.set_major_locator(ticker.NullLocator())
        ax[i][0].yaxis.set_major_locator(ticker.NullLocator())

        ax[i][1].imshow(hsd_vis_list[i],cmap="gray")
        ax[i][1].xaxis.set_major_locator(ticker.NullLocator())
        ax[i][1].yaxis.set_major_locator(ticker.NullLocator())

        ax[i][2].imshow(label_vis_list[i])
        ax[i][2].xaxis.set_major_locator(ticker.NullLocator())
        ax[i][2].yaxis.set_major_locator(ticker.NullLocator())
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return fig

    


def split_transform_all(dtms_path_all, labels_path_all,hsds_path_all,
                        transform_list,transform_name_list,
                        chipsize= 512,overlap=64,
                        write=False,output_dir=None,vis=True):
    """
    this functions apply split and transform to dtm, hsd and labels of multiple sites at once.
    Write out the output as png is optional
    It first grab all the corresponding hsd and label, then call split_transform to apply augmentation site by site.
    If write out the subimages and the transformed subimages,
    the output is saved in this following structure:
        - output_root
        |--siteName1
        |--|--dtms/*
        |--|--hsds/*
        |--|--labels/*
        |--siteName2
        ...
    """
    
    data_preprocessed_dict  = {}
    if vis:
        plot_dir = os.path.abspath(os.path.join(os.getcwd(),"transformation_plots"))
        os.makedirs(plot_dir,exist_ok=True)
        example_idx_dict = {'ardmayle':9,"knockainey":35,"kilbixy":45}

    if write:
        os.makedirs(output_dir,exist_ok=True)
    for i in range(len(dtms_path_all)):
        # for a given hillshade file name, find its corresponding label and dtm file, as well as siteName
        hsd_path = hsds_path_all[i]
        site = re.findall(siteRegex, os.path.basename(hsd_path))[0]
        r = re.compile(site,re.IGNORECASE)
        label_path = list(filter(r.search, labels_path_all)) [0]
        dtm_path = list(filter(r.search, dtms_path_all)) [0]
        siteName = site.lower()
        data_preprocessed_dict[siteName] = {}
        if write:
            output_dir_site = os.path.join(output_dir ,siteName)
            os.makedirs(output_dir_site,exist_ok=True)
        else:
            output_dir_site = None

        print(f"Processing {siteName} ...")
        transformed_result  = split_transform(hsd_path,label_path,dtm_path,siteName,
                            chipsize=chipsize,overlap=overlap,
                            transform_list=transform_list,transform_name_list=transform_name_list,
                            write =write,output_root= output_dir_site )

        subdtm_augmented_all,subhsd_augmented_all,sublabel_augmented_all,subcoords_augmented_all = unpack_transformed_result(transformed_result)
        # add these into prepared data dictionary after the key that correponds to the siteName
        data_preprocessed_dict[siteName]['dtm'] = subdtm_augmented_all 
        data_preprocessed_dict[siteName]['hsd'] = subhsd_augmented_all 
        data_preprocessed_dict[siteName]['label'] = sublabel_augmented_all 
        data_preprocessed_dict[siteName]['coord_list'] = subcoords_augmented_all

        
        if transform_list!=None and vis: # visualize transformation results
            fig = visualize_transformations(transformed_result,title=[s for s in transformed_result.keys()],example_idx=example_idx_dict[siteName])
            plotname = f"{siteName}transformation.png"
            fig.savefig(os.path.join(plot_dir, plotname))
    return data_preprocessed_dict




if __name__=="__main__":

    # if not importing this script, split with overlap and write results

    root_path= os.path.abspath(os.path.join(os.getcwd(),"../"))
    raw_data_default = os.path.join(root_path,"00_Data_Preprocessed")
    output_root_default = os.path.join(root_path,"01_Data_Prepared")
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-in',"--input_dir",nargs="?",default=raw_data_default,
                        help='input data directory, including the dtm, hsd and lables of all the sites that need to be preprocessed')
    parser.add_argument('-out',"--output_dir",nargs="?",default=output_root_default,
                        help='output directory, where the spitted and augmented data will be stored')
    parser.add_argument('-c','--chip_size', nargs="?",default=512,type=int, 
                        help="select chip size for subimages")
    parser.add_argument('-o','--overlap', nargs="?",default=64,type=int, 
                        help="select amount of overlap between subimages")
    args = parser.parse_args()

    chipsize = args.chip_size
    overlap= args.overlap
    raw_data = args.input_dir
    output_root = args.output_dir
    os.makedirs(output_root,exist_ok=True)

    tifs = glob(os.path.join(raw_data,"*.tif"))
    # get all the dtm, labels and hillshade files underneath 00_Data_Preprocessed
    dtms_path_all = list(filter(re.compile(".*dtm.tif$").search,tifs))
    labels_path_all = list(filter(re.compile(".*label.tif$").search,tifs))
    hsds_path_all = list(filter(re.compile(".*\\d+_\\d+.*.tif$").search,tifs))

    #site_list = ["ardmayle","kilbixy","knockainey"]
    transform_list = [flip("horizontal"), flip("veritical"),rotate(90),rotate(180),rotate(270)]
    transform_name_list = ["hoflip","verflip","rot90","rot180","rot270"]
    data_preprocessed_dict = split_transform_all(dtms_path_all, labels_path_all,hsds_path_all,
                                                transform_list,transform_name_list,
                                                chipsize= chipsize,overlap = overlap,
                                                write=True,output_dir=output_root)


        
