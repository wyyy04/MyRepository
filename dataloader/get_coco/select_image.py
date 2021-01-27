from readtxt import loadDataset
import os
import shutil

#从COCO训练集中选取motivations_clean中训练和测试使用的所有图片

rdir='D:\download\\train2014'#源目录
odir='D:\data'#目标目录

data = loadDataset()
data = data[:,0]
print(data)

for im_name in data:
    print(im_name) #文件名
    r = os.path.join(rdir,im_name)
    o = os.path.join(odir,im_name) #得到源文件&目标文件完整目录
    print(r,o)
    shutil.copy(r,o)  # 复制文件到目标路径；移动move
