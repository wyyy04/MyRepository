import numpy as np
import pandas as pd
import os
import cv2 as cv
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

DataDir = 'loader\data\\'
Datasetfile = DataDir + 'dataset.txt'
Embeddingfile = DataDir + 'skipthoughts.npz'
Clusterfile = DataDir + 'clusters.npz'
Clustersnamefile = DataDir + 'clustersname.txt'
Clusterfile_256 = DataDir + 'clusters_256.npz'
ImageDir = DataDir + 'COCO_motivations_clean\\'


class COCO_Dataloader():
    def __init__(self):
        self.inputsize = 64     # the size input to model(chennel_num*size*size)
        self.class_num = 256    # class number
        self.textloader = Read_text()
        self.npzloader = Read_npz()
        self.traindata, self.testdata,self.testlabels = self.load_data()

    def get_dataset(self):
        return self.traindata, self.testdata
    def get_testset(self):
        return self.testdata
    def get_test_labels(self):
        return self.testlabels


    def load_data(self):
        img_name = self.textloader.loadDataset_IMGname().squeeze(-1)
        trainIndex, testIndex = self.textloader.loadDataset_TrainTest_Index()

        labels = self.npzloader.loadMotivationClusters()
        trainlabel = labels[trainIndex]
        testlabel = labels[testIndex]
        trainlabel = torch.from_numpy(trainlabel)
        testlabel = torch.from_numpy(testlabel)

        train_img_name = img_name[trainIndex]
        test_img_name = img_name[testIndex]

        traindata = torch.Tensor(len(trainlabel),3,self.inputsize, self.inputsize)
        testdata = torch.Tensor(len(testlabel),3,self.inputsize, self.inputsize)

        train_transform = transforms.Compose([
            transforms.Resize((self.inputsize, self.inputsize)),  # 缩放
            transforms.RandomCrop(self.inputsize, padding=4),  # 随机裁剪
            transforms.ToTensor(),  # 图片转张量，同时归一化0-255 ---》 0-1
            transforms.Normalize((0.3948,0.3928,0.3890), (0.2725,0.2715,0.2715)),
            # [0.39481035 0.39288068 0.3890493][0.27250803 0.27149644 0.27156308]
        ])


        for i,pic_name in enumerate(train_img_name):
            print("Loading Training image No.",i+1,"/",len(train_img_name),"  :",pic_name)
            img = Image.open(ImageDir + pic_name).convert('RGB')  # 读取图像
            img = train_transform(img)
            traindata[i] = img
            if i>30:break


        for i,pic_name in enumerate(test_img_name):
            print("Loading Testing image No.",i+1,"/",len(test_img_name),"  :",pic_name)
            img = Image.open(ImageDir + pic_name).convert('RGB')  # 读取图像
            img = train_transform(img)
            testdata[i]=img
            print(img)
            if i > 0: break

        trdata = torch.utils.data.TensorDataset(traindata, trainlabel.long())
        tedata = torch.utils.data.TensorDataset(testdata, testlabel.long())

        return trdata,tedata,labels[testIndex]








class Read_npz():
    def __init__(self):
        self.a = 0

    # 读取npz
    def loadnpzData(self,file):
        data = np.load(file)
        return data

    # 读取skipthoughts.npz
    def loadMotivationEmbedding(self):
        file = Embeddingfile
        embedding = self.loadnpzData(file)
        data = np.array(embedding['m'])
        # (10191, 4800)
        return data

    def loadMotivationClusters(self):  # labels
        file = Clusterfile
        embedding = self.loadnpzData(file)
        data = np.array(embedding['result'])
        # (10191,)
        return data



class Read_text():
    def __init__(self):
        self.a = 0

    # 读取txt到list
    def loadtxt(self,file):
        f = open(file, 'r')
        sourceInLine = f.readlines()
        dataset = []
        for line in sourceInLine:
            temp1 = line.strip('\n')
            temp2 = temp1.split('\t')
            dataset.append(temp2)
        f.close()
        return dataset

    # 读取dataset.txt
    def loadDataset(self):
        file = Datasetfile
        dataset = self.loadtxt(file)
        dataset = np.array(dataset)
        return dataset

    # 读取clustersname.txt
    def loadClustersname(self):
        file = Clustersnamefile
        dataset = self.loadtxt(file)
        dataset = np.array(dataset)
        return dataset

    def loadDataset_i_m(self):
        data = self.loadDataset()
        data = pd.DataFrame(data)
        data.columns = ['image', 'action', 'motivation', 'scene', 'traintest']

        data = np.array(data)
        return data



    def loadDataset_TrainTest_Index(self):
        data = self.loadDataset()
        data = pd.DataFrame(data)
        data.columns = ['image', 'action', 'motivation', 'scene', 'traintest']

        trainIndex = [i for i, x in enumerate(list(data['traintest'])) if x == 'train']
        testIndex = [i for i, x in enumerate(list(data['traintest'])) if x == 'test']

        return trainIndex, testIndex


    def loadDataset_IMGname(self):
        data = self.loadDataset()
        data = pd.DataFrame(data)
        data.columns = ['image', 'action', 'motivation', 'scene', 'traintest']

        img = data.loc[:, ['image']]
        img = np.array(img)

        return img







if __name__ == "__main__":

    #
    # print("")
    # r = Read_text()
    #
    # img_name = r.loadDataset_IMGname().squeeze(-1)
    # trainIndex,testIndex = r.loadDataset_TrainTest_Index()
    #
    #
    # train_img_name = img_name[trainIndex]
    # test_img_name = img_name[testIndex]
    #
    #
    # pic_name = train_img_name[0]
    # img = cv.imread(ImageDir + pic_name)


    L = COCO_Dataloader()
    # L.load_data()
