import numpy as np
cluster_num = 3

img = np.arange(5*7)
img = img.reshape([5,7])
label = np.random.randint(1, cluster_num + 1, size=[5,7])
index = np.where(label == 1)
print(label,index[1])

print(img)
print(img - 2)
im_l = img[index]
print(im_l)
