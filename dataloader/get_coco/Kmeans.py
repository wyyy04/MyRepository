import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from readnpz import loadMotivationEmbedding
from readtxt import loadDataset_i_m
import time
from data_path import *



# 读取数据
data = loadMotivationEmbedding()
dataname = loadDataset_i_m()
# 聚类数量
k = 256
# 训练模型
t = time.time()
model = KMeans(n_clusters=k)
model.fit(data)
# 分类中心点坐标
centers = model.cluster_centers_
# 预测结果
result = model.predict(data)

print("分类用时：  ", (time.time() - t)%60 ,'min')
print(result)
np.savez(Clusterfile, result=result)


f_name = open(Clustersnamefile, 'w')


# centerembedding = []
for i in range(len(centers)):
    center = centers[i]
    print(i, " :  ", center)
    data = np.array(data)
    center = np.array(center)
    centerindex = np.argmin((np.sum((data - center) ** 2, axis=1)) ** 0.5)#欧氏距离
    # print(data[centerdata])
    # centerembedding.append(data[centerindex])
    mtype = dataname[centerindex, 1]
    f_name.write(str(i) + '\t' + mtype + '\n')

f_name.close()


