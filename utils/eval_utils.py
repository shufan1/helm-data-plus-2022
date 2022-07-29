import numpy as np
import torch
from torchmetrics import JaccardIndex


def accuracy(y_pred,y_true):
    assert type(y_pred) == type(y_true)
    correct_pixels = y_pred==y_true
    try: # if type tensor, conver to numpy, np.ndarray is easier for plotting
        correct_pixels = correct_pixels.numpy() 
        
    except:
        pass
    return(np.sum(correct_pixels)/np.size(correct_pixels))

        
def get_mask_per_class(mask_array,num_classes=3):
    # index 0: background mask
    mask_per_class_list = []
    for i in range(num_classes):
        class_i_mask = mask_array== i
        mask_per_class_list.append(class_i_mask)
    return mask_per_class_list


def getTPFPFNTN_mask (mask_pred,mask_true):
    # we really dislike high false negative: this means missing an actual feature
    tp_mask = (mask_pred==1) & (mask_true==1).astype(int)
    fn_mask = (mask_pred==0) & (mask_true==1).astype(int)
    tn_mask = (mask_pred==0) & (mask_true==0).astype(int)
    fp_mask = (mask_pred==1) & (mask_true==0).astype(int)
    return tp_mask,fn_mask,tn_mask,fp_mask


def count_tpfntnfp(tp_mask,fn_mask,tn_mask,fp_mask):
    return np.sum(tp_mask),np.sum(fn_mask) ,np.sum(tn_mask) ,np.sum(fp_mask) 


def tpfntnfp_rate(tp_count,fn_count,tn_count,fp_count):
    tpr = tp_count/(tp_count+fn_count)
    fpr = fp_count/(tn_count+fp_count)
    # false negative rate = 1-tpr
    # true negative rate = 1-fpr
    return tpr,1-tpr,1-fpr,fpr


def getTPFNTNFP_all(y_pred,y_true,num_classes=3):
    mask_pred_list = get_mask_per_class(y_pred)
    mask_true_list = get_mask_per_class(y_true)

    # index 0: background class
    masks_collection = {i:{} for i in range(num_classes+1)}
    tpfpfntn_counts =np.zeros((num_classes+1,4))
    tpfpfntn_rates =np.zeros((num_classes+1,4))

    for i in range(num_classes+1):
        mask_pred = mask_pred_list[i].numpy()
        mask_true = mask_true_list[i].numpy()
        tp_mask,fn_mask,tn_mask,fp_mask = getTPFPFNTN_mask(mask_pred,mask_true)
        masks_collection[i]['tp'] = tp_mask
        masks_collection[i]['fn'] = fn_mask
        masks_collection[i]['tn'] = tn_mask
        masks_collection[i]['fp'] = fp_mask

        tpfpfntn_counts[i,:] =  count_tpfntnfp(tp_mask,fn_mask,tn_mask,fp_mask)
        tpfpfntn_rates[i,:] = tpfntnfp_rate(*tpfpfntn_counts[i,:])

    return tpfpfntn_rates,tpfpfntn_counts,masks_collection

def get_mIoU(y_pred, y_true,num_classes,device):
    
    if y_true.dtype != 'torch.LongTensor':
        y_true = y_true.long()
    if y_pred.dtype != 'torch.LongTensor':
        y_pred = y_pred.long()
        
    y_pred, y_true = y_pred.to(device),y_true.to(device)
        
    mIoU = JaccardIndex(num_classes=num_classes).to(device)
    mIoU = mIoU(y_pred, y_true).item()
    return mIoU

def get_IoU(y_pred,y_true,class_idx):
    pred_inds = y_pred==class_idx
    target_inds = y_true ==class_idx

    if target_inds.sum().item() == 0:
        return np.nan
    else:
        intersection = pred_inds[target_inds].sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        iou = intersection/union
        return iou
