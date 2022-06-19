"""# Imports """
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras 
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint

"""# Load the Data"""

x_train = np.load('dataset_25thousands/x_train.npy')
x_test = np.load('dataset_25thousands/x_test.npy')
y_train = np.load('dataset_25thousands/y_train.npy')
y_test = np.load('dataset_25thousands/y_test.npy')
class_names = ['cow', 'panda', 'lion', 'tiger', 'raccoon', 'monkey', 'hedgehog', 'zebra', 'horse',
              'owl', 'elephant', 'squirrel', 'sheep', 'dog', 'bear', 'kangaroo', 'whale', 'crocodile',
              'rhinoceros', 'penguin', 'camel', 'flamingo', 'giraffe', 'pig', 'cat']
num_classes = len(class_names)

# print(num_classes)
# print(len(x_train))
# print(x_train.shape)

"""Show some random data """

idx = randint(0, len(x_train))
plt.show(x_train[idx].reshape(28,28),cmap='gray')   
print(class_names[int(y_train[idx].item())])
#print(x_train[idx])

"""# Preprocess the Data """


# Reshape and normalize
image_size = 28
x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1).astype('float32')

x_train /= 255.0
x_test /= 255.0

# Convert class vectors to class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""# CNN Model """

# Define model
model = keras.Sequential()
model.add(layers.Convolution2D(16, (3, 3),
                        padding='same',
                        input_shape=x_train.shape[1:], activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Convolution2D(32, (3, 3), padding='same', activation= 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Convolution2D(64, (3, 3), padding='same', activation= 'relu'))
model.add(layers.MaxPooling2D(pool_size =(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(25, activation='softmax')) 

# Optimizers
adam = tf.optimizers.Adam()
sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.0001, nesterov=False)
rms = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0)

# Metrics
from keras.metrics import top_k_categorical_accuracy
def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)    
def acc_top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
def acc_top4(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=4)
def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer= adam,
              metrics=['acc', acc_top2, acc_top3, acc_top4, acc_top5])

print(model.summary())
keras.utils.plot_model(model, "model_strcture.png", show_shapes=True)


"""# Training """

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
# model.fit(x = x_train, y = y_train, validation_split=0.1, batch_size = 256, callbacks=callback, verbose=2, epochs=60)
model.fit(x_train, y_train, validation_split=0.1, batch_size = 256, verbose=2, epochs=25)

"""# Testing """

score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuarcy(top_k=1): {:0.2f}%'.format(score[1] * 100))
print('Test accuarcy(top_k=3): {:0.2f}%'.format(score[3] * 100))
print('Test accuarcy(top_k=5): {:0.2f}%'.format(score[5] * 100))

"""# Inference """

idx = randint(0, len(x_test))
img = x_test[idx]
plt.show(img.squeeze()) 
pred = model.predict(np.expand_dims(img, axis=0))[0]
ind = (-pred).argsort()[:3]
latex = [class_names[x] for x in ind]
print(latex)