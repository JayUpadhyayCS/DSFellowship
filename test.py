# Tensorflow / Keras
from tensorflow import keras # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__) # print version
from keras.models import Sequential # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras.layers import Masking,Bidirectional, LSTM, RepeatVector, Dense, TimeDistributed,MaxPooling1D, Flatten, Conv1D,Conv2D,Dropout, MaxPooling2D # for creating layers inside the Neural Network
from keras.optimizers import Adam , SGD
# Data manipulation
import pandas as pd # for data manipulation
print('pandas: %s' % pd.__version__) # print version
import numpy as np # for data manipulation
print('numpy: %s' % np.__version__) # print version
import numpy.ma as ma
# Sklearn
import sklearn
print('sklearn: %s' % sklearn.__version__) # print version
from sklearn.preprocessing import MinMaxScaler # for feature scaling
from sklearn.model_selection import train_test_split
# Visualization
import plotly 
import plotly.express as px
import plotly.graph_objects as go
print('plotly: %s' % plotly.__version__) # print version
from matplotlib import pyplot as plt

#file accessing
import os
# time stuff
from datetime import timedelta
import calendar




def shaping2(datain,datain2, timestep):
    #print(arr)
    cnt=0
    for row in range(len(datain2.index)): #picks a row at every iteration, allows to reduction of input and inclusion of multiple time series, remove start and step to run on full dataset 
    # Convert input dataframe to array and flatten
        #datain.iloc[row].fillna(datain.iloc[row].mean).to_numpy().flatten()

        if datain2.index[row] not in datain.index:
            row+=1 # CAN REMOVE
            continue
        
        
        arr=datain.iloc[row].to_numpy().flatten() # flatten row
        arr=np.where(np.isnan(arr), ma.array(arr, mask=np.isnan(arr)).mean(), arr) 
        arr2=datain2.iloc[row].to_numpy().flatten()
        arr3=np.concatenate((arr,arr2)).reshape(2,110)
        
        
        for mth in range(0, len(datain2.columns)-(2*timestep)+1): # Define range lenght of the dates - 2* amount of timesep?? +1
            cnt=cnt+1 # Gives us the number of samples. Later used to reshape the data
            X_start=mth # Start month for inputs of each sample
            X_end=mth+timestep # End month for inputs of each sample
            Y_start=mth+timestep # Start month for targets of each sample. Note, start is inclusive and end is exclusive, that's why X_end and Y_start is the same number
            Y_end=mth+2*timestep # End month for targets of each sample.  
            
            # Assemble input and target arrays containing all samples
            if cnt==1:
                X_comb=arr3[X_start:X_end]
                Y_comb=arr3[0][Y_start:Y_end]
            else: 
                X_comb=np.append(X_comb, arr3[:,X_start:X_end])
                Y_comb=np.append(Y_comb, arr3[:0,Y_start:Y_end])
    
    # Reshape input and target arrays
    X_out=np.reshape(X_comb, (cnt, timestep, 2))
    Y_out=np.reshape(Y_comb, (cnt, timestep, 2))
    return X_out, Y_out

arr1=np.array([1,2,3,4,5,6, 7,8,9])
#print(arr.shape)
#arr1=arr1.reshape(3,3,1)
#print(arr)
#print(arr.shape)
arr2=np.array([10,11,12,13,14,15, 16,17,18])
#arr2=arr2.reshape(3,3,1)
#
print ('arr1:',arr1)
#
print ('arr2:',arr2)
#arr3=np.concatenate((arr1,arr2))
# arr3=arr3.resize(3,3,2)
arr3=np.concatenate((arr1,arr2)).reshape(2,9)

print(arr3[:1,1:5])
# arr3=arr3.reshape(3,3,2)
print(arr3) #sart
X_train=[]
Y_train=[]
for i in range(3,9):
    X_train.append(arr3[i-3:i])
    Y_train.append(arr3[0][i])
print(X_train)
print(Y_train)
shaping2(arr1,arr2,)
X_train_shape=np.reshape(X_train,(3,3,2))
print(X_train)
#end
# print(arr3.shape)
# arr3=arr3.reshape(3,3,2)
# print(arr3)
# print(arr3.shape)


#Notess
# Can put them in a dataframe and then work from then.

#Foudn online shaping function
# # convert history into inputs and outputs
# def to_supervised(train, n_input, n_out=7):
# 	# flatten data
# 	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
# 	X, y = list(), list()
# 	in_start = 0
# 	# step over the entire history one time step at a time
# 	for _ in range(len(data)):
# 		# define the end of the input sequence
# 		in_end = in_start + n_input
# 		out_end = in_end + n_out
# 		# ensure we have enough data for this instance
# 		if out_end <= len(data):
# 			X.append(data[in_start:in_end, :])
# 			y.append(data[in_end:out_end, 0])
# 		# move along one time step
# 		in_start += 1
# 	return array(X), array(y)

def lstm_data_transform(x_data, y_data, num_steps=5):
    """ Changes data to the format for LSTM training 
for sliding window approach """
    # Prepare the list for the transformed data
    X, y = list(), list()
    # Loop of the entire data set
    for i in range(x_data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps
        # if index is larger than the size of the dataset, we stop
        if end_ix >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = x_data[i:end_ix]
        # Get only the last element of the sequency for y
        seq_y = y_data[end_ix]
        # Append the list with sequencies
        X.append(seq_X)
        y.append(seq_y)
    # Make final arrays
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array, y_array