import numpy as np
from tensorflow.keras import layers
from tensorflow import keras 
import tensorflow as tf

def padding(x): # x.shape = (n, 784)
    padding_x_group = []
    print(len(x))
    for k in range(len(x)):
        img_data = x[k]
        img_data = img_data.reshape(28,28)
        padding_data = np.full((32,32), 255, dtype = 'uint8')
        for i in range(28):
            for j in range(28):
                padding_data[i+2][j+2] = img_data[i][j]
        padding_data = padding_data.reshape(1,-1)
        padding_data = padding_data.squeeze()
        #print(padding_data.shape)
        padding_x_group.append(padding_data)
    padding_x = np.array(padding_x_group, dtype = 'uint8')
    print(padding_x.shape)
    return padding_x