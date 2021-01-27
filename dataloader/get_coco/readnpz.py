import numpy as np
from data_path import *

#读取npz
def loadnpzData(file):
    data = np.load(file)
    return data

#读取skipthoughts.npz
def loadMotivationEmbedding():
    file = Embeddingfile
    embedding = loadnpzData(file)
    data = np.array(embedding['m'])
    # (10191, 4800)
    return data

def loadMotivationClusters():
    file = Clusterfile
    embedding = loadnpzData(file)
    data = np.array(embedding['result'])
    #(10191,)
    return data

