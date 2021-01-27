import numpy as np
from readnpz import loadMotivationClusters
from readtxt import loadDataset


dataset = loadDataset()
result = loadMotivationClusters()

annoted_data = []
for i in range(len(dataset)):
    annoted_data.append([dataset[i][0],result[i],dataset[i][4]])

np.savez('annoted_data.npz', image = dataset[:,0], train_or_test = dataset[:,4], type_num  = result )

# data = np.load('annoted_data.npz')
# print(data['image'],data['train_or_test'],data['type_num'])
