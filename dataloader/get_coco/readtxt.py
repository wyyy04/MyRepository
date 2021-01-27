import numpy as np
import pandas as pd
from data_path import *





#读取txt到list
def loadtxt(file):
    f=open(file,'r')
    sourceInLine=f.readlines()
    dataset=[]
    for line in sourceInLine:
        temp1=line.strip('\n')
        temp2=temp1.split('\t')
        dataset.append(temp2)
    f.close()
    return dataset

#读取dataset.txt
def loadDataset():
    file = Datasetfile
    dataset = loadtxt(file)
    dataset = np.array(dataset)
    return dataset

#读取clustersname.txt
def loadClustersname():
    file = Clustersnamefile
    dataset = loadtxt(file)
    dataset = np.array(dataset)
    return dataset



def loadDataset_i_m():
    data = loadDataset()
    data = pd.DataFrame(data)
    data.columns = ['image', 'action', 'motivation', 'scene', 'traintest']

    data = np.array(data)
    return data




def loadDataset_TrainTest():
    data = loadDataset()
    data = pd.DataFrame(data)
    data.columns = ['image', 'action', 'motivation', 'scene', 'traintest']

    data_train = data[data['traintest'] == 'train']
    data_test = data[data['traintest'] == 'test']
    data_train = data_train.loc[:,['image','motivation']]
    data_test = data_test.loc[:, ['image', 'motivation']]

    data_train = np.array(data_train)
    data_test = np.array(data_test)
    #(7665, 2)
    #(2526, 2)
    return data_train,data_test


