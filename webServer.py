

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:35:51 2021

@author: User-1
"""

import os
import sys

from focal_loss import BinaryFocalLoss
import os
import sys
import argparse
import numpy as np
from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D,AveragePooling1D
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.layers.wrappers import Bidirectional, TimeDistributed
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.models import Model
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from tensorflow.keras import initializers
from tensorflow.keras.layers import Activation, Dense, Add
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn.metrics import auc
from tensorflow.keras.layers import LSTM
from focal_loss import BinaryFocalLoss
from Bio import SeqIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import StratifiedKFold

def get_model():


    input_shape = (81,4)
    inputs = Input(shape = input_shape)
    convLayer = Conv1D(filters = 16, kernel_size = 7,activation = 'relu',kernel_regularizer = regularizers.l2(1e-5), bias_regularizer = regularizers.l2(1e-4),input_shape = input_shape)(inputs)
    poolingLayer = MaxPooling1D(pool_size = 2, strides=2)(convLayer)
    dropoutLayer = Dropout(0.5)(poolingLayer)
    convLayer2 = Conv1D(filters = 32, kernel_size = 5,activation = 'relu',kernel_regularizer = regularizers.l2(1e-4), bias_regularizer = regularizers.l2(1e-5))(dropoutLayer)
    poolingLayer2 = MaxPooling1D(pool_size = 2, strides=2)(convLayer2)
    dropoutLayer2 = Dropout(0.25)(poolingLayer2)
    convLayer3 = Conv1D(filters = 64, kernel_size = 5,activation = 'relu',kernel_regularizer = regularizers.l2(1e-4), bias_regularizer = regularizers.l2(1e-5))(dropoutLayer2)
    poolingLayer3 = MaxPooling1D(pool_size = 2, strides=2)(convLayer3)
    dropoutLayer3 = Dropout(0.25)(poolingLayer3)
    flattenLayer = Flatten()(dropoutLayer3)
    dropoutLayer4 = Dropout(0.25)(flattenLayer)
    denseLayer2 = Dense(100, activation = 'relu',kernel_regularizer = regularizers.l2(1e-4),bias_regularizer = regularizers.l2(1e-4))(dropoutLayer4)
    outLayer = Dense(1, activation='sigmoid')(denseLayer2)
    model2 = Model(inputs = inputs, outputs = outLayer)
    model2.compile(loss='binary_crossentropy',optimizer= 'adam', metrics=['binary_accuracy']);
	
    return model2


modelProMN = get_model()
modelProMN.load_weights('prom70-CNN_weights.h5') #That i already submitted
#modelProMN.load_weights('D:/Research papers/Busan Conference Paper/Prom70 Performance/Prom70-OriginalNeg-Weights.h5')
import numpy as np

def encode_seq(s):
    Encode = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}
    return np.array([Encode[x] for x in s])


X1 = {}
accumulator=0

Strng=["AGATTTAATGTTTTACTGTTGGAACGAGATGTTTTGTTAGTGCCCTAAATTATATAAAATATATATTTATTTTTAAATTATAGAATATTTTTATAGGTCACTTCCAACCTGGCGTTATGCAGAAGCACAAATCTGATCCAAAATCCAAACAGTCTTCCCGATCAAAACACCTATGGGCTTGGACTTGTTTGTACACATATAAATCTAACTTAATCCATATAGAGAAAGTTAAACTGAAATTTATAATTTAA",
       "GAACAGGAGGGAGCATGGAGTGCACTTCTTGTTCTAGTATATTGAGGCCTCGTTTGGTAGAGGCTCCATGATTCTCTAATACAGTGATTCTGAGTGATTTTCTATTGCAAGTGAATCTATTTGACGAAAACTGTTTGATAAATAGGCTGTGAAGTGATTTTTGAAGGATTAAAGAGTGAGAAGCAGGTTGAGAGTGGTGGGAAGCAGGTTTTTTTGCTCCCAATTTCTAGTACAAAGTAGAGACTAGATTC",
       "GGGGCGCGGCGGCCAACTGCCTTGCCCTTGCACTGATGGATGCCGGGACCCTAGTCCCGAAGACGGATGGGTTTGGGCGTTTGGCTGGGAGCAGGATGCACGGGAGCCACTCGTTCGGTCGTTCCTGCGCCGCGATGCAGATCTACTCCACATCTACACTATTCTTTATCAATACCATTCATTGTAGTCTCTCACTGTCAGCGTCGTCAAGGCCTCTCCCTACTTTTCTTTCTTTTTTTTAACCCCTATGG"]
       
def Prom70(Strng):
    p=0
    n=0
    prediction=""
    X1 = {}
    my_hottie = encode_seq((Strng))
    out_final=my_hottie
     # out_final=out_final.astype(int)
    out_final = np.array(out_final)
    X1[accumulator]=out_final
      #out_final=list(out_final)
    X1[accumulator] = out_final    
    X1 = list(X1.items()) 
    an_array = np.array(X1)
    an_array=an_array[:,1]    
    transpose = an_array.T
    transpose_list = transpose.tolist()
    X1=np.transpose(transpose_list)
    X1=np.transpose(X1)
    pr=modelProMN.predict(X1)
    pr=pr.round()
    if(pr==1):
        prediction='Query Sequence is Sigma70 promoter'
        p=1
    else:
        prediction='Query Sequence is non promoter'
        n=1
    return prediction,p,n

predictions=[]
def predict_seq(seqs):
    for i in range(len(seqs)):
        pred=Prom70(seqs[i])
        predictions.append(pred)
    return predictions


Testsequences = [] 
for record in SeqIO.parse("D:/Research papers/Busan Conference Paper/Prom70 Performance/Prom70-CNN Shujaat/test_DatasetSigma70.txt", "fasta"):
    Testsequences.append(record.seq.upper())
 
#Testsequences=Testsequences[2500:2860]    
#%%     
#predict_seq(Testsequences)
tp=0
fn=0     
for i in range(len(Testsequences)):
    e=Prom70(Testsequences[i])
    print(e[0])
