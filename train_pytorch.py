import os
from argparse import ArgumentParser

import matplotlib.pyplot  as  plt
import torch
from datanew import getM4dataset
from torch import optim
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
from AFDmodel import  NbeatsNet
from torch.utils.data import Dataset
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_metrics(y_true,y_hat):
    error=np.mean(np.square(y_true-y_hat))
    smape=np.mean(2*np.abs(y_true-y_hat)/(np.abs(y_true)+np.abs(y_hat)))
    return smape
def  get_script_arguments():
    parser=ArgumentParser(description='N-Beats')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--disable-plot', action='store_true', help='Disable interactive plots')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()
    
def split(arr,size):
    arrays=[]
    while len(arr)>size:
        slice_=arr[:size]
        arrays.apppend(slice_)
        arr=arr[size:]
    arrays.append(arr)
    return arrays
def train_model(net,optimizer,trainl):
    net.train()
    smape=AverageMeter()
    for  temp in tqdm(trainl):
        optimizer.zero_grad()
        
        train_x,train_y,real_t,imag_t=temp[0].float().cuda(),temp[1].float().cuda(),temp[2].float().cuda(),temp[3].float().cuda()
        real_t=real_t.view(-1)
        imag_t=imag_t.view(-1)
        output_trainx,output_trainy=net(train_x,real_t,imag_t)
        #print("outputtrain:",output_train)
        #print("train_Y:",train_y)
        loss=F.mse_loss(output_trainy,train_y)
        loss.backward()
        optimizer.step()
        smapenew=get_metrics(output_trainy.detach().cpu().numpy(),train_y.detach().cpu().numpy())
        smape.update(smapenew.item(),train_y.size(0))
        #print("2020新年快乐")
    return smape.avg

def val_model(net,optimizer,valloader):
    net.eval()
    smape=AverageMeter()
    optimizer.zero_grad()
    for temp in tqdm(valloader):
        print("123456")
        val_x,val_y,real_t,imag_t=temp[0].float().cuda(),temp[1].float().cuda(),temp[2].float().cuda(),temp[3].float().cuda()
        real_t=real_t.view(-1)
        imag_t=imag_t.view(-1)
        output_valx,output_valy=net(val_x,real_t,imag_t)
        loss=F.mse_loss(output_valy,val_y)
        smapenew=get_metrics(output_valy.detach().cpu().numpy(),val_y.detach().cpu().numpy())
        smape.update(smapenew.item(),val_y.size(0))
        print("2020happynewyear")
    return smape.avg


def main():
    #CHECKPOINT_NAME = 'nbeats.th'
    x_train=pd.read_csv("/home/lfn/Nbeatsd2h/data/train/Weekly-train.csv")
    x_test=pd.read_csv("/home/lfn/Nbeatsd2h/data/val/Weekly-test.csv")
    x_afd=pd.read_csv("/home/lfn/Nbeatsd2h/data/train/trainweek.csv")
    device = torch.device("cuda:0")
    device_ids = [0]
    lr = 1e-3
    n_epochs = 100
    bs = 128
    forecast_length = 13
    backcast_length = 2*forecast_length
    trainset=getM4dataset(backcast_length,forecast_length,dataframe_train=x_train,dataframe_test=x_test,dataframe_AFD=x_afd,thetas_dim=10,is_train=True,transform=None)
    train_loader=torch.utils.data.DataLoader(trainset,batch_size=bs,shuffle=True,num_workers=0)
    valset=getM4dataset(backcast_length,forecast_length,dataframe_train=x_train,dataframe_test=x_test,dataframe_AFD=x_afd,thetas_dim=10,is_train=False,transform=None)
    val_loader=torch.utils.data.DataLoader(valset,batch_size=bs,shuffle=False,num_workers=0)
    args=get_script_arguments()

    net = NbeatsNet(device=device,
                    stack_types=[NbeatsNet.GENERIC_BLOCK, NbeatsNet.SEASONALITY_BLOCK],
                    forecast_length=forecast_length,
                    thetas_dims=[2, 8],
                    nb_blocks_per_stack=3,
                    backcast_length=backcast_length,
                    hidden_layer_units=1024,
                    share_weights_in_stack=False)
    net.cuda()
    #def plot_model(x, target, grad_step):
    #    if not args.disable_plot:
    #       print('plot()')
    #        plot(net, x, target, backcast_length, forecast_length, grad_step)
    optimizer=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-5)
    best_smape=100
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=16,gamma=0.66)
    net=nn.DataParallel(net,device_ids=device_ids)
    for epoch in range(n_epochs):
        print('lr:', scheduler.get_lr()[0])
        start_time=time.time()
        train_smape=train_model(net,optimizer,train_loader)
        val_smape=val_model(net,optimizer,val_loader)
        elapsed_time=time.time()-start_time
        print('Epoch {}/{} \t train_smape={:.4f} \t val_smape={:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, train_smape, val_smape, elapsed_time))
        torch.save(net.state_dict(), './ckpt/'+str(epoch)+'_'+str(val_smape)+'.pt')
        if val_smape<best_smape:
            best_smape=val_smape
            torch.save(net.state_dict(), './ckpt/best_'+str(epoch)+'_'+str(best_smape)+'.pt')
        scheduler.step()
        print("*********best_smape**********",best_smape)

    
if __name__ == '__main__':
    main()
