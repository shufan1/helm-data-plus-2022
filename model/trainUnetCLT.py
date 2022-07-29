"""
Run training as a command line tool. Same steps and output as trainUnet.ipynb
Eg: 
    python trainUnetCLT.py -d 02_Data_Augmented_2class -train ardmayle kilbixy -test knockainey -use_h -prop 0.8 -s 512 -lr 0.0007 -m 0.8 -e 5 -b 10 -f -o train_ard_kilb
    this means: train on samples from site ardmayle, kilbixy
                test on site knocakiney
                use hillshade as model input
                train - val split is 80% VS 20%
                subimge size = 512
                training parameters: learning rate =  0.0007, epoch = 35, batch size = 10 and momentum = 0.8
                do not freeze encoder block when training
                do not load any previous weights: -p path_to_weight_file if needed
                and description for the model is: train_ard_kilb. 
                This makes all the output file names follow: Unet_train_ard_kilb_... follwed by training parameter description.
                                                            e.g.: UnetDice_train_ard_kilb_1ep7e-04lr10b8e-01m.pth
"""

import os
import time 
import numpy as np
from glob import glob
import argparse
import re
import json
import gc
import segmentation_models_pytorch as smp
from PIL import Image
import cv2 as cv2
import time
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm,tnrange
from utils import vis_utils,data_utils,eval_utils
from utils.model_utils import eval_model,train_Unet
from torchmetrics import JaccardIndex,F1Score
from utils.preprocess_utils import split_w_overlap, augmentDataset, flip, rotate
from preprocess.preparedata_01 import split_transform_all
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# use GPU if availabe
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  
print("device being used: ",device)

# helper function to visualize if needed
def visualize(dtm_tensor=None,hsd_tensor=None, gt_mask_tensor=None,pred_mask_tensor=None,
          plot_titles = ['dtm','hillshade','ground truth','prediction']):

    """
    visualize a sample, which type of image to plot is optional
    """
    tensor2img = torchvision.transforms.ToPILImage()
    plot_list = [dtm_tensor,hsd_tensor,gt_mask_tensor,pred_mask_tensor]
    plot_list = [i for i in plot_list if (not i is None)]

    fig,ax = plt.subplots(1,len(plot_list),figsize=(len(plot_list)*5,5))
    for i  in range(len(plot_list)):
        if plot_list[i].shape[0] == 3:
          plot_tensor = tensor2img(plot_list[i])
        else:
          plot_tensor = plot_list[i].squeeze(0)
        if plot_titles[i] == "image" or plot_titles[i] == "hillshade" :
            ax[i].imshow(plot_tensor,cmap='gray')
        else:
            ax[i].imshow(plot_tensor)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].set_title(plot_titles[i],fontsize=16)
    
    return fig


def get_class_weights(dataset,num_classes=2):
    pixel_counts = torch.ones(num_classes)
    for i in range(len(dataset)):
        _,label = dataset[i]
        for j in range(num_classes):
            pixel_counts[j] += torch.sum(label ==j)
    class_weights  = 1/pixel_counts/sum(1/pixel_counts)
    return class_weights

def unpack_dict(data_preprocessed_dict,sitelist):
    subdtms = []
    subhsds = []
    sublabels  = []
    coord_list = []
    for site in sitelist:
        subdtms.extend(data_preprocessed_dict[site]['dtm'])
        subhsds.extend(data_preprocessed_dict[site]['hsd'])
        sublabels.extend(data_preprocessed_dict[site]['label'])
        coord_list.extend(data_preprocessed_dict[site]['coord_list'])
    return subdtms,subhsds,sublabels,coord_list
        


def setup_and_train(model,train_set,val_set,num_classes,
                lr,momentum,batch_size,epochs,model_folder,model_name,
                freeze_encoder,load_weights,weight_path=None):
    
    # load training and validation data with Pytorch Dataloader, this retrive samples by batch, for smaller computational memory
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, sampler=None,)
    validloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, sampler=None,)

    if load_weights:
        print("loading weight_path")
        try:
            model= torch.load(weight_path,map_location=torch.device(device))
        except:
            model.load_state_dict(torch.load(weight_path),map_location=torch.device(device))
    else:
        model.to(device)
        
    if freeze_encoder:
        print("freeze encoder")
        for parma in model._modules['encoder'].parameters():
            parma.requires_grad = False
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)   
    print(" number of trainable parameters: %d"%pytorch_total_params)

    # declare pytorch optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),lr=lr, momentum=momentum)
    criterion = smp.utils.losses.DiceLoss() 
    thresh = 0.5
    # call tran_Unet to start the training process
    start_time = time.time()
    model,train_epoch_loss,train_epoch_iou,val_epoch_loss,val_epoch_iou = train_Unet(
        model,model_name,trainloader,num_classes,
        criterion,optimizer,epochs,thresh,device,validloader)
    end_time = time.time()
    print("Finished training, total time: %.3fmin"%((end_time-start_time)/60))

    # save trained model
    lr_str = '{:.0e}'.format(lr)
    momentum_str = '{:.0e}'.format(momentum)
    weights_filename = f"Unet{model_name}_{epochs}ep{lr_str}lr{batch_size}b{momentum_str}m"

    weights_file= weights_filename  +".pth"
    modelpath = os.path.join(model_folder, weights_file)
    torch.save(model, modelpath)

    # save train progress
    train_progress = dict({"train_loss": train_epoch_loss,
                    "train_mIoU": train_epoch_iou,
                    "val_loss": val_epoch_loss, 
                    "val_iou": val_epoch_iou})
    progressfilename = f"{weights_filename}_progress.json"
    progressfilepath = os.path.join(model_folder,progressfilename)
    with open(progressfilepath, "w") as f:
        json.dump(train_progress, f, indent=4)

    return model,modelpath,train_epoch_loss,train_epoch_iou,val_epoch_loss, val_epoch_iou



def load_and_train(dtms_path_all,labels_path_all,hsds_path_all,
                trainSites,testSites,model_folder,model_name,subimage_size=512,overlap=64,
                train_prop=0.8,num_classes = 2, batch_size = 10, 
                lr = 0.0007,epoch=15, momentum=0.8,
                use_hillshade=True,drop_edge =True,
                freeze_encoder = False, load_weights = False, weight_path = None):
    """This is the main function for processing and training"""

    #1.load data, split and augmentation
    #site_list = ["ardmayle","kilbixy","knockainey"]
    print("Read dtm, hillshade and label of each site, apply spliting and augmentation")
    transform_list = [flip("horizontal"), flip("veritical"),rotate(90),rotate(180),rotate(270)]
    transform_name_list = ["hoflip","verflip","rot90","rot180","rot270"]
    data_preprocessed_dict = split_transform_all(dtms_path_all,labels_path_all,hsds_path_all,
                                            transform_list,transform_name_list,
                                            chipsize = subimage_size, overlap = overlap,
                                            write=False,vis=False)
    #------------------------------------------------------------------------------------------------------------
    # 2. unnpack prepared data and make train, val and test set
    print("Creating  train, val and test set")
    # duplicate the channel gray scale image 2 times -> 3 channel img
    gray2rgb_transform = transforms.Compose([data_utils.gray2rgb])
    pad_transform = transforms.Compose([
            transforms.ToTensor(),
            data_utils.pad4sides((subimage_size,subimage_size))])

    # get training data based on trainiing site name(s)
    subdtms_train,subhsds_train,sublabels_train,coordlist_train = unpack_dict(data_preprocessed_dict,trainSites)
    trainValDataset =  data_utils.HELMDataset(subdtm_arrays=subdtms_train,subhsd_arrays=subhsds_train,
                                            sublabel_arrays=sublabels_train,coord_list=coordlist_train,
                                            use_hillshade = use_hillshade,drop_edge =drop_edge, size = subimage_size,
                                            transform=gray2rgb_transform,pad_transform=pad_transform)
    print("Train val set sample size:",len(trainValDataset))
    # do train and val split
    train_set,val_set = data_utils.split_data(trainValDataset,train_prop,seed =seed)

    # get test data based on test site name(s)
    subdtms_test,subhsds_test,sublabels_test,coordlist_test = unpack_dict(data_preprocessed_dict,testSites)
    test_set =  data_utils.HELMDataset(subdtm_arrays=subdtms_test,subhsd_arrays=subhsds_test,
                                            sublabel_arrays=sublabels_test,coord_list=coordlist_test,
                                            use_hillshade = use_hillshade,drop_edge =drop_edge, size = subimage_size,
                                            transform=gray2rgb_transform,pad_transform=pad_transform)
    print("Test set sample size:",len(trainValDataset))

    for i in range(3):
        X,y = train_set[np.random.randint(0,len(train_set))]
        fig = visualize(X,y,plot_titles=["Hillshade (input)","Label"])
    
    #------------------------------------------------------------------------------------------------------------
    # 3. Load U-Net and pre-trained weight, and train

    UnetVGG16 = smp.Unet('vgg16',classes=1,encoder_weights='imagenet',activation = "sigmoid")
    params_encoder = sum(p.numel() for p in UnetVGG16 ._modules['encoder'].parameters())
    print("Number of parameters in Encoder block: %d"%params_encoder)
    params_encoder = sum(p.numel() for p in UnetVGG16 ._modules['decoder'].parameters())
    print("Number of parameters in decoder block: %d"%params_encoder)
    params_encoder = sum(p.numel() for p in UnetVGG16 ._modules['segmentation_head'].parameters())
    print("Number of parameters in segmentation head: %d"%params_encoder)

    # define and create the folder to save trained weights
    model_folder = os.path.join(root_path,"model/models_weights_temp")
    os.makedirs(model_folder,exist_ok=True)

    # load training and validation data with Pytorch Dataloader, this retrive samples by batch, for smaller computational memory
    model,modelpath,train_epoch_loss,train_epoch_iou,val_epoch_loss, val_epoch_iou = setup_and_train(
                UnetVGG16,train_set,val_set,num_classes,
                lr,momentum,batch_size,epochs,model_folder,model_name,
                freeze_encoder,load_weights,weight_path=None)
    
    # plot training progress and save the plot
    lr_str = '{:.0e}'.format(lr)
    momentum_str = '{:.0e}'.format(momentum)
    weights_filename = f"Unet{model_name}_{epochs}ep{lr_str}lr{batch_size}b{momentum_str}m"
    fig,ax = plt.subplots(1,2,figsize=(16,5))
    ax[0].plot(np.arange(1,len(train_epoch_loss)+1),train_epoch_loss,label="train loss")
    ax[0].plot(np.arange(1,len(train_epoch_loss)+1),val_epoch_loss,label="validation loss")
    ax[0].set_xlabel("Epoch",fontsize=16)
    ax[0].set_ylabel("Dice loss", fontsize=16)
    ax[0].legend()
    ax[1].plot(np.arange(1,len(train_epoch_loss)+1),train_epoch_iou,label="train mIoU")
    ax[1].plot(np.arange(1,len(train_epoch_loss)+1),val_epoch_iou,label="validation mIoU")
    ax[1].set_xlabel("Epoch",fontsize=16)
    ax[1].set_ylabel("IoU", fontsize=16)
    ax[1].legend()
    lr_str = '{:.0e}'.format(lr)
    momentum_str = '{:.0e}'.format(momentum)
    plotname= f"trainprogress{weights_filename}.png"
    #print(os.path.join(models_folder,plotname))
    plt.savefig(os.path.join(model_folder,plotname))
    plt.show()

    gc.collect() 
    torch.cuda.empty_cache()

    #------------------------------------------------------------------------------------------------------------
    # 4 Testset evaluation (preliminary) using thershold of 0.5
    print("Predicting on new site(s)")
    thresh = 0.
    criterion = smp.utils.losses.DiceLoss() 
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, sampler=None,)
    y_prob_list,loss,IoU = eval_model(model,testloader,criterion,num_classes,thresh,device)

    # compute F1 and IoU
    # reshape the list of y_prob
    # reshape the list of y_prob [N H W] where N = number of sample
    y_predprob = torch.stack(y_prob_list, dim=0)
    # get all ground truthlabel
    y = [ pad_transform(sublabel.copy()) for sublabel in test_set.sublabels]
    y = torch.stack(y).squeeze(1)

    target = torch.Tensor(y).to(device).bool()
    pred = torch.Tensor(y_predprob)
    Jaccard = JaccardIndex(num_classes=2,threshold=0.5).to(device)
    IoU = Jaccard(pred,target)
    F1_score = F1Score(num_classes=1,threshold=0.5).to(device)
    pred_flatten = torch.flatten(pred,0)
    target_flatten = torch.flatten(target,0)
    F1 = F1_score(pred_flatten,target_flatten)
    print(f"Iou = {round(IoU.item(),4)}, F1 =  {round(F1.item(),4)},")

    #plot a few examples    
    N = 4
    fig,ax = plt.subplots(int(N/2),4,figsize=(4*3,N/2*3))
    for n in range(N):
        model.eval()
        X,y = train_set[np.random.randint(0,20)]
        y_hat = model(X.unsqueeze(0).to(device))
        row = int(n%(N/2))
        col = int(n//(N/2))
        ax[row][col*2].imshow(y.squeeze(0))
        ax[row][col*2+1].imshow(y_hat.detach().cpu().squeeze(0).squeeze(0).numpy())
        # set axes  off
        ax[row][col*2].xaxis.set_major_locator(ticker.NullLocator())
        ax[row][col*2].yaxis.set_major_locator(ticker.NullLocator())
        ax[row][col*2+1].xaxis.set_major_locator(ticker.NullLocator())
        ax[row][col*2+1].yaxis.set_major_locator(ticker.NullLocator())

        ax[0][0].set_title("ground truth")
        ax[0][1].set_title("predicted probability")
        ax[0][2].set_title("ground truth")
        ax[0][3].set_title("predicted probability")
        fig.tight_layout() 
        fig.savefig(f"{weights_filename}test_predprob.png")
    fig

    #------------------------------------------------------------------------------------------------------------
    

if __name__=="__main__":

    """
    python trainUnetCLT.py -d 00_Data_Preprocessed -train ardmayle kilbixy -test knockainey -use_h -prop 0.8 -s 512 -lr 0.0007 -m 0.8 -e 5 -b 10 -f -o train_ard_kilb

    """
    # define data path
    root_path = os.path.abspath(os.path.join(os.getcwd(),"../"))
    data_root = os.path.join(root_path,"00_Data_Preprocessed")

    trainSites = ["ardmayle","kilbixy"]
    testSites = ['knockainey']
    chip_size  = 512
    train_prop = 0.8
    seed = 42
    lr = 0.0007# change it to 0.005 
    momentum = 0.8
    epochs = 10
    batch_size = 10
    freeze_encoder = False
    load_weights = False
    num_classes = 2

    parser = argparse.ArgumentParser(description='Train Unet on HLEM data')
    parser.add_argument('-d','--data_root',nargs="?",default = data_root,help="input data root")
    parser.add_argument('-train',nargs='+',default=trainSites,
                        help="training set site name(s) list")
    parser.add_argument('-test',nargs='+',default=testSites,
                        help="test set site name(s) list")
    parser.add_argument('-prop',nargs="?",default =train_prop,type=float,
                    help="training val split, train proportion 0-1")
    parser.add_argument('-s','--size',nargs="?",default = chip_size,type=int,
                        help="img size")
    parser.add_argument('-use_h','--hillshade',default=False,action="store_true",help="use hillshade as input")
    parser.add_argument('-lr','--lr',nargs="?",default = lr,help="learning rate",type=float)
    parser.add_argument('-m','--momentum',nargs="?",default = momentum,help="momentum",type=float)
    parser.add_argument('-e','--epoch',nargs="?",default = epochs,help="epochs",type=int)
    parser.add_argument('-b','--batch_size',nargs="?",default = batch_size,help="batch size",type=int)
    parser.add_argument('-f','--freeze',default=False,action="store_true",help="freeze encoder weights")
    parser.add_argument('-l','--load',default=False,action="store_true",help="load pretrained weights")
    parser.add_argument('-p','--path',nargs="?",default="",help="path to pretrianed weights that will be loaded before training")
    parser.add_argument('-o','--model_name',nargs="?",default="",help="customized name for the  saved model file")

    args = parser.parse_args()
    data_root = args.data_root
    trainSites = args.train
    testSites = args.test
    train_prop = args.prop
    lr,momentum,epochs,batch_size = args.lr,args.momentum,args.epoch,args.batch_size
    freeze_encoder,load_weights,use_hillshade = args.freeze, args.load, args.hillshade
    weight_path = args.path
    model_name = args.model_name


    print(data_root,trainSites,testSites,train_prop,chip_size)
    print(f"learning parameters: lr= {lr}, momentum = {momentum}, epoch = {epochs}, batch size = {batch_size}") 
    print(weight_path,freeze_encoder,load_weights,model_name)


    root_path = os.path.abspath(os.getcwd())
    raw_data = os.path.join(root_path,"00_Data_Preprocessed")

    tifs = glob(os.path.join(raw_data,"*.tif"))
    # get all the dtm, labels and hillshade files underneath 00_Data_Preprocessed
    dtms_path_all = list(filter(re.compile(".*dtm.tif$").search,tifs))
    labels_path_all = list(filter(re.compile(".*label.tif$").search,tifs))
    hsds_path_all = list(filter(re.compile(".*\\d+_\\d+.*.tif$").search,tifs))

    # create folder to save weigths and other results
    model_folder = os.path.join(root_path,"model/models_weights_temp")
    os.makedirs(model_folder,exist_ok=True)

    
    load_and_train(dtms_path_all,labels_path_all,hsds_path_all,
                trainSites,testSites,model_folder,model_name,subimage_size=512,overlap=64,
                train_prop=0.8,num_classes = 2, batch_size = 10, 
                lr = 0.0007,epoch=15, momentum=0.8,
                use_hillshade=True,drop_edge =True,
                freeze_encoder = False, load_weights = False, weight_path = None)
