# [CS420] Machine Learning: Free-hand Sketch Classification Problem
## Introduction
This is SJTU machine learning course group project code. The content of this project is free-hand sketch classification problem. The data we used in this project is a part of data from [The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset).

## Datatools  
``datatools`` folder includes `vec2pix.py` and `padding.py` these two files.  
`vec2pix.py` transforms the sequentially recorded data in the source dataset into vector images in SVG format, and then into PNG images. Finally, gray sampling is carried out to obtain a 28\*28 matrix to represent an image.  
`padding.py` fills in the 28\*28 image. It fills blank pixels around it to make it a 32\*32 image, which is easy to use as input for the pre-training model.

## Models
``models`` folder includes `classification_CNN.py` and `classification_pretrained_model.py` these two files.  
`classification_CNN.py` is the CNN model built by us manually, including three convolutional layers, pooling layers and two fully connected layers.
In `classification_pretrained_model.py`, we used two pre-trained models in `tensorflow.keras.application`: `MobileNet_v2` and `ResNet50`.
