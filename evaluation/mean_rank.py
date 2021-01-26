import pandas as pd
import numpy as np

def evaluate(score,ground_truth):
    score = np.array((score))
    ground_truth = np.array(ground_truth)
    img_num,moti_num = score.shape
    score = pd.DataFrame(score)
    score = score.rank(ascending=False,method='first', axis=1)
    score = np.array(score)
    res = score[range(img_num), ground_truth]
    res = np.median(res)
    return res

img_num = 10191
moti_num = 256

score = np.random.random([img_num,moti_num])
truth = np.random.randint(0,moti_num,[img_num])
print("函数结果：",evaluate(score,truth))

print("模型预测结果：\n",score)
print("真值：\n",truth)
score = pd.DataFrame(score)
score = score.rank(method='first',axis=1)
print("排名结果：\n",score)
score = np.array(score)
res = score[range(img_num),truth]
print("真值排名名次：\n",res)
res = np.median(res)
print("真值排名中位数：",res)

