import re
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import json
import shutil
import sys
import argparse


siteRegex = re.compile("Kilbixy|Fore|Glenogra|Ardmayle|Kilmacahill|Ballynahnich|Knockainey", re.IGNORECASE)
class2num  = {"background":0,"historical_walls":1,"modern_features":2}
num2class = {0:"background",1:"historical_walls",2:"modern_features"}

def order_images_labels(img_fullpaths,label_fullpaths):
    # order img files and label files so they correspond to each other
    img_files_list = []
    label_files_list = []
    for img_file in img_fullpaths:
        siteName = re.findall(siteRegex,os.path.basename(img_file))[0]
        r = rf".+{siteName}.+"
        r = re.compile(r,re.IGNORECASE)
        label_file = list(filter(r.match, label_fullpaths)) [0]
        img_files_list.append(img_file)
        label_files_list.append(label_file)
    return img_files_list,label_files_list



def writeidx2classDict(class_list):
    # ask user to order class from 0 to len(classlist)-1 after viewing the label of each class
    # this mapping from num to class name is saved as a json file
    confirm = False
    while not confirm: # keep asking until confirmed
        s = len(class_list)-1
        class_list = ['background','historical_walls','modern_features']
        class_ordered= input(
            f"map {str(class_list)} to [0,1,2]...\n Type feature class name in the order of the previous plot seperated by ',' no space between (e.g. modern_features,historical_walls,background):")
        class_ordered = list(map(str,class_ordered.split(",")))

        idx_in_uniqueNum2class= dict({ i:class_ordered[i] for i in range(len(class_ordered))})
        idx2class_json = {"class2idx":idx_in_uniqueNum2class}
        json_object = json.dumps(idx2class_json, indent = 4) 
        print(json_object)
        confirm = input(f"Confirm? (type Yes/No): ") == "Yes"
 
    return idx_in_uniqueNum2class



def checkclass_single(img,label,output_label_folder,class_list):
        # examine each map and its labeled label, and their dimensions
        siteName = re.findall(siteRegex,os.path.basename(img))[0]
        print("\nSite:", siteName)
        img_array = np.asarray(Image.open(img))
        print("img array shape:", img_array.shape)
        label_array = np.asarray(Image.open(label))
        label_array = label_array.copy()
        print("label array shape:", label_array.shape)

        fig,ax = plt.subplots(1,2,figsize=(14,7))
        ax[0].imshow(img_array,cmap="gray")
        ax[1].imshow(label_array,cmap="gray")
        plt.show()

        # plot each class, examine and assign each class number its corresponding feature name
        unique_class = np.unique(label_array)
        
        print("unique num found",str(unique_class))
        fig,ax = plt.subplots(1,len(unique_class),figsize=(20,7))
        for i in range(len(unique_class)):
            ax[i].imshow(label_array==unique_class[i],cmap="gray")
            ax[i].set_title("class %d"%i)
            # encode each unique number with new class id: 0 ,1 2,3...
            label_array[label_array==unique_class[i]] = float(i)
        plt.show()

        confirm_ordering  = False
        while not confirm_ordering:
            label_array_copy = label_array.copy()
            idx_in_uniqueNum2class = writeidx2classDict(class_list)

            for i in range(len(unique_class)):
                # encode each unique number with new class id: 0 ,1 2,3...
                label_array_copy[label_array==i] =  float(class2num[idx_in_uniqueNum2class[i]])
            
            fig,ax = plt.subplots(1,len(unique_class),figsize=(20,7))
            for i in range(len(unique_class)):
                classname = num2class[i]
                ax[i].imshow(label_array_copy==i,cmap="gray")
                ax[i].set_title(f"class {i}: {classname}")
            plt.show()

            confirm_ordering = input(f"Confirm? (type Yes/No): ") == "Yes"

        label_array = label_array_copy

        # save the processed label to outputpath
        output_label_file  = os.path.join(output_label_folder ,f"{siteName}_label.tif")
        im = Image.fromarray(label_array)
        im.save(output_label_file )#os.path.join(processed_folder ,output_label_file))


def checkclass_all(img_folder,label_folder,output_folder,class_list):

    img_fullpaths = glob(os.path.join(img_folder,'*.tif'))
    r = re.compile("^((?!dtm).)*$")
    img_fullpaths = list(filter(r.search,img_fullpaths))
    label_fullpaths = glob(os.path.join(label_folder,'*label*.tif'))
    img_files_list,label_files_list = order_images_labels(img_fullpaths,label_fullpaths)
    
    for i in range(len(img_files_list)):
        img = img_files_list[i]
        label = label_files_list[i]
        checkclass_single(img,label,output_folder,class_list)
 



if __name__=="__main__":
    root_path = os.path.abspath(os.path.join(os.getcwd() ,"../../"))
    datafolder= os.path.join(root_path,"ArcGis/Projects/HELM/data/Azavea/data")
    img_folder_default = os.path.join(datafolder,'data')
    label_folder_default = os.path.join(datafolder,"extracted_labels_shp")
    ouput_dir_default = os.path.join(datafolder,"Azavea_processed")
   
   
    parser = argparse.ArgumentParser(description='Preprocess data exported from ArcGis')
    parser.add_argument('-d','--img_dir',nargs='?',default=img_folder_default,
                        help="full path to the folder that stores all the img")
    parser.add_argument('-l','--label_dir',nargs='?',default=label_folder_default,
                        help="full path to the folder that stores all the labels")
    parser.add_argument('-o','--output_dir',nargs="?",default =ouput_dir_default,
                        help="full path to the output folder")
    args = parser.parse_args()

    img_folder = args.img_dir
    label_folder = args.label_dir
    output_folder = args.output_dir
    os.makedirs(output_folder,exist_ok=True)
    print("output saved in ",output_folder)

    class_list = ['background','historical_walls','modern_features']

    checkclass_all(img_folder,label_folder,output_folder,class_list)




