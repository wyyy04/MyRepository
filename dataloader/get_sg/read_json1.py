import json
import numpy as np

dic = 'C:\\Users\\26458\\Desktop\\custom_data_info.json'

# 读取json文件内容,返回字典格式
with open(dic,'r',encoding='utf8')as fp:
    json_data = json.load(fp)

    data = json_data

for key in data.keys():

    print(key,":",len(data[key]),"组,  大小是",np.array(data[key]).shape)
    print("        ",data[key])

