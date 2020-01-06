import csv
import torch as t
#from AFD import AFD_test
import numpy as np
import torch.nn.init as init
import pandas as pd
import os
from torch.utils.data import Dataset
import math
#from compiler.ast import flatten

class getM4dataset(Dataset):
    def __init__(self,backcast_length,forecast_length,dataframe_train,dataframe_test,dataframe_AFD,thetas_dim,is_train=True,transform=None):
        self.backcast_length=backcast_length
        self.forecast_length=forecast_length
        self.dftrain =dataframe_train
        self.dftest  =dataframe_test
        self.dfafd   =dataframe_AFD
        self.dim=thetas_dim
        self.transform=transform
        self.is_train=is_train
    def __len__(self):
        return len(self.dftrain)
    def __getitem__(self,idx):
        label = np.zeros(self.forecast_length)
        if self.is_train:
            timeseries_train=self.dftrain.ix[idx]
            timeseries_train=timeseries_train[1:]
            timeseries_train=np.array(timeseries_train)
            time_series_cleaned=[float(s) for s in timeseries_train if s==s]
            time_series_cleaned_forlearning_x=np.zeros(self.backcast_length)
            time_series_cleaned_forlearning_y=np.zeros(self.forecast_length)
            j=np.random.randint(self.backcast_length,len(time_series_cleaned)+1-self.forecast_length)
            time_series_cleaned_forlearning_x=time_series_cleaned[j-self.backcast_length:j]
            time_series_cleaned_forlearning_y=time_series_cleaned[j:j+self.forecast_length]
            #thetas=AFD_test(time_series_cleaned)
            thetas=self.dfafd.ix[idx]
            #thetas=thetas.view(-1)
            #print("time_series_cleaned_forlearning_x:",time_series_cleaned_forlearning_x)
            #print("time_series_cleaned_forlearning_y:",time_series_cleaned_forlearning_y)
            thetas_new=[complex(s[1:-1]) for s in thetas if s==s]
            
            real_t=np.real(thetas_new)

            imag_t=np.imag(thetas_new)

            time_series_cleaned_forlearning_x=np.array(time_series_cleaned_forlearning_x)
            time_series_cleaned_forlearning_y=np.array(time_series_cleaned_forlearning_y)
            real_t=np.array(real_t)
            imag_t=np.array(imag_t)
            print(time_series_cleaned_forlearning_x)
            #print("real_t:",real_t)
            #print("imag_t:",imag_t)
            #print("ssssssssss")
            #print(time_series_cleaned_forlearning_x)
            temp=[time_series_cleaned_forlearning_x, time_series_cleaned_forlearning_y]
            #priint("temptrain:",temp)
            return temp
        else :
            timeseries_test_x=self.dftrain.ix[idx]
            timeseries_test_x=timeseries_test_x[1:]
            timeseries_test_x=np.array(timeseries_test_x)
            time_series_cleaned_x=[float(s) for s in timeseries_test_x if s==s]

            timeseries_test_y=self.dftest.ix[idx]

            timeseries_test_y=timeseries_test_y[1:]
            time_series_cleaned_y=[float(s) for s in timeseries_test_y if s==s]
            test_y=np.zeros(self.forecast_length)
            #timeseries_test_y=np.array(timeseries_test_y)
            #thetas=AFD_test(time_series_cleaned_x,self.dim)
            thetas=self.dfafd.ix[idx]
            thetas_new=[complex(s[1:-1]) for s in thetas if s==s]
            real_t=np.real(thetas_new)
            imag_t=np.imag(thetas_new)
            
            test_x=time_series_cleaned_x[-self.backcast_length:]
            test_y=time_series_cleaned_y[0:self.forecast_length]
            test_x=np.array(test_x)
            test_y=np.array(test_y)
            temp=[test_x,test_y]
            #print("temp:",temp)
            return temp 
        
