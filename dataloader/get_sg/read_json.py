import json
import numpy as np

dic = 'C:\\Users\\26458\\Desktop\\custom_prediction.json'

# 读取json文件内容,返回字典格式
with open(dic,'r',encoding='utf8')as fp:
    json_data = json.load(fp)
    # print('这是文件中的json数据：',json_data)
    # print("['0']:",len(json_data['0']))
    # print("['0']:", json_data['0'].keys())
    # print('这是读取到文件数据的数据类型：', type(json_data))
    data = json_data['0']

for key in data.keys():

    print(key,":",len(data[key]),"组,  大小是",np.array(data[key]).shape)

# print(data['bbox'])
# print(data['bbox_labels'])
# print(data['bbox_scores'])


# print("\n\n",np.reshape(data['rel_pairs'],[-1]).max())

print(data['rel_scores'])