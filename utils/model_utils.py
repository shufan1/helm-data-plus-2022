import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import time
from utils.eval_utils import get_mIoU
from torchmetrics import JaccardIndex


def train_Unet(model,model_name,trainloader,num_classes,criterion,optimizer,epochs,thresh,device,validloader=None):
    model.to(device)
    if device.type != 'cpu':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
    jaccard = JaccardIndex(num_classes,threshold=thresh).to(device)
    use_val = validloader != None
    # store traning progress
    train_epoch_loss = []
    train_epoch_iou = []
    val_epoch_loss = []
    val_epoch_iou = []
    val_iou_max = 0.0

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss =[]
        running_iou = []
        with tqdm(trainloader, unit="batch") as batch:
            for X,y in batch:      
                # x: image to predict on
                # y: predicated segmentation label from model                    
                batch.set_description(f"Epoch {epoch+1}")
                model.train() # set model to training mode
                X,y = X.to(device),y.to(device)         
                #  zero the parameter gradients
                #  PyTorch accumulates the gradients on subsequent backward passes: loss.backward()
                optimizer.zero_grad()            
                # forward + backward + optimize
                if device.type != 'cpu':
                    torch.cuda.synchronize()
                    start.record()
                    y_hat = model(X)
                    end.record()
                    torch.cuda.synchronize()
                    forward_time = round(start.elapsed_time(end)*0.001,3)  # seconds
                else:
                    start = time.time()
                    y_hat = model(X)
                    end = time.time()
                    forward_time = round(end-start,3)# seconds
                
                y = y.type(torch.torch.float32)
                # remove the channel dimension of y
                # dim of y: [batch_size, 1, height, width] -> dim of y: [batch_size, height, width]
                y = torch.squeeze(y, 1)
                # same for predicted output from the model
                y_hat = torch.squeeze(y_hat, 1)
                loss = criterion(y_hat, y)

                if device.type != 'cpu':
                    start.record()
                    loss.backward() # dloss/da for every parameter a which has requires_grad=True. update a.grad
                    end.record()
                    torch.cuda.synchronize()
                    backward_time = round(start.elapsed_time(end)*0.001,3)  # seconds
                else:
                    start = time.time()
                    loss.backward()
                    end = time.time()
                    backward_time = round(end-start,3)# seconds
                optimizer.step() # updates the value of parmaeter a using the gradient a.grad. a += -lr * a.grad                
                
                IoU_train = jaccard(y_hat,y.bool())
                running_loss.append(loss.item())
                running_iou.append(IoU_train.item())
                
                # print training progress
                batch.set_postfix_str(
                    f"train_loss={round(loss.item(),4)}, train_mIoU = {round(IoU_train.item(),3)}, forward {forward_time}s, backward {backward_time}s.")#,epoch_running_loss = running_loss)#, accuracy=100. * accuracy)
            
            if epoch %10 ==0: # save mode every 10 epoch
                torch.save(model.state_dict(), f'.{model_name}_{epoch}ep.pth')

            # if epoch > 20: # if want to change learning rate after X epochs
            #         for param_group in optimizer.param_groups:
            #             param_group['lr'] = 0.000
            loss_curr_epoch = np.mean(running_loss)
            iou_curr_epoch = np.mean(running_iou)
            train_epoch_loss.append(loss_curr_epoch)
            train_epoch_iou.append(iou_curr_epoch)

            if use_val:
                # evaluate current model on validation dataset
                y_predprob,val_loss,val_iou = eval_model(model,validloader,criterion,num_classes,thresh,device)
                val_epoch_loss.append(val_loss)
                val_epoch_iou.append(val_iou)
                if val_iou > val_iou_max:
                    torch.save(model.state_dict(), f'./{model_name}_currbest.pth')
                tqdm.write("Epoch train loss = %.3f, train mIoU = %.3f, validation loss = %.3f, mIoU = %.3f "%(loss_curr_epoch,iou_curr_epoch,val_loss,val_iou))
            else:
                tqdm.write("Epoch train loss = %.3f, train mIoU = %.3f"%(loss_curr_epoch,iou_curr_epoch))

    if use_val:
        return model,train_epoch_loss,train_epoch_iou,val_epoch_loss,val_epoch_iou
    else:
        return model,train_epoch_loss,train_epoch_iou 



def eval_model(model,dataloader,criterion,num_classes,thresh,device):
    jaccard = JaccardIndex(2,threshold=thresh).to(device)
    model.to(device)
    model.eval() # set model to evaluation mode
    y_proba_list = []
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as batch:
            loss_list = []
            IoU_list = []
            for X,y in batch:

                batch.set_description("Evaluating")     
                X, y = X.to(device),y.to(device)
                # compute output
                y_hat = model(X)
                # remove the channel dimension of y
                # dim of y: [batch_size, 1, height, width] -> [batch_size, height, width]
                y = y.type(torch.torch.float32)
                y = torch.squeeze(y, 1)
                y_hat = torch.squeeze(y_hat, 1)
                y_proba_list.extend(y_hat)
                loss = criterion(y_hat, y)
                IoU = jaccard(y_hat,y.bool())
                loss_list.append(loss.item())
                IoU_list.append(IoU.item())
                batch.set_postfix(loss = round(loss.item(),3),meanIoU = round(IoU.item(),3))

    return y_proba_list,np.mean(loss_list),np.mean(IoU_list)


def compute_loss(model,X,y,criterion):
    #  X and y are both tensor
    if X.dim()!=4: #this happens when only one sample in X
        X = torch.unsqueeze(X, 0)
    y_hat = model(X)
    y = y.type(torch.LongTensor)
    if y.dim() ==4:
        y = torch.squeeze(y, 1)  
    #criterion = nn.CrossEntropyLoss()
    loss = criterion(y_hat, y)
    return loss.item()
    
    
