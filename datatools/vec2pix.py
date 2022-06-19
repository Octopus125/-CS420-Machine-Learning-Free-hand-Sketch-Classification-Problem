import numpy as np
from six.moves import xrange
import six
import svgwrite  # conda install -c omnia svgwrite=1.1.6
import os
import tensorflow.compat.v1 as tf
import glob
from PIL import Image, ImageDraw
import re
import shutil
import cairosvg

def get_bounds(data):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0])
        y = float(data[i, 1])
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)

def draw_strokes(data, svg_filename='sample.svg', width=48, margin=1.5, color='black'):
    """ convert sequence data to svg format """
    min_x, max_x, min_y, max_y = get_bounds(data)
    if max_x - min_x > max_y - min_y:
        norm = max_x - min_x
        border_y = (norm - (max_y - min_y)) * 0.5
        border_x = 0
    else:
        norm = max_y - min_y
        border_x = (norm - (max_x - min_x)) * 0.5
        border_y = 0
  
    # normalize data
    norm = max(norm, 10e-6)
    scale = (width - 2*margin) / norm
    dx = 0 - min_x + border_x
    dy = 0 - min_y + border_y
  
    abs_x = (0 + dx) * scale + margin
    abs_y = (0 + dy) * scale + margin
  
    # start converting
    dwg = svgwrite.Drawing(svg_filename, size=(width,width))
    dwg.add(dwg.rect(insert=(0, 0), size=(width,width),fill='white'))
    lift_pen = 1
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in xrange(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i,0]) * scale
        y = float(data[i,1]) * scale
        lift_pen = data[i, 2]
        p += command+str(x)+","+str(y)+" "
    the_color = color  # "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
    dwg.save()

def vec2pix(vec):
    draw_strokes(vec, svg_path) # draw svg
        
    fileHandle = open(svg_path) # open svg
    svg = fileHandle.read()
    fileHandle.close()
        
    exportFileHandle = open(png_path, 'w') # open png
    cairosvg.svg2png(bytestring=svg, write_to=png_path, output_width=28, output_height=28) # draw png
    exportFileHandle.close()
        
    img = Image.open(png_path, 'r').convert('L')  # covert to grayscale
    img_data = np.array(img)
    img_data = img_data.reshape(1,-1)
    img_data = img_data.squeeze()
    return img_data

classes = ["cow", "panda", "lion", "tiger", "raccoon", "monkey", "hedgehog", "zebra", "horse", "owl",
           "elephant", "squirrel", "sheep", "dog", "bear", "kangaroo", "whale", "crocodile", "rhinoceros",
           "penguin", "camel", "flamingo", "giraffe", "pig","cat"]
svg_path = "temp.svg"
png_path = "temp.png"


# convert vec to pix for each category
for animal in classes:
    npz_path = "../data/sketchrnn_" + animal + ".npz"
    npz_data = np.load(npz_path, encoding='latin1', allow_pickle=True) 
    print(animal)
    print(npz_data.files)
    npz_data_test = npz_data['test']
    npz_data_train = npz_data['train']
    npz_data_valid = npz_data['valid']
    
    npy_data_test_group = []
    npy_data_train_group = []
    
    for test in npz_data_test:
        img = vec2pix(test)
        npy_data_test_group.append(img)
        
    for train in npz_data_train:
        img = vec2pix(train)
        npy_data_train_group.append(img)
        
    for valid in npz_data_valid:
        img = vec2pix(valid)
        npy_data_train_group.append(img)
        
    npy_data_test = np.array(npy_data_test_group)
    npy_data_train = np.array(npy_data_train_group)
    print(npy_data_train.shape)
    print(type(npy_data_train))
    
    test_path = "/amax/data/hys/other/quickdraw/npy/" + animal + "_test.npy"
    train_path = "/amax/data/hys/other/quickdraw/npy/" + animal + "_train.npy"
    np.save(test_path, npy_data_test)
    np.save(train_path, npy_data_train)

x_train = np.empty([0, 784])
y_train = np.empty([0])

x_test = np.empty([0, 784])
y_test = np.empty([0])

# combine all the data in 25 categories
for idx in range(len(classes)):
    animal = classes[idx]
    test_path = "/amax/data/hys/other/quickdraw/npy/" + animal + "_test.npy"
    train_path = "/amax/data/hys/other/quickdraw/npy/" + animal + "_train.npy"
    
    train_data = np.load(train_path)
    train_labels = np.full(train_data.shape[0], idx)
    
    test_data = np.load(test_path)
    test_labels = np.full(test_data.shape[0], idx)
    
    x_train = np.concatenate((x_train, train_data), axis=0)
    y_train = np.append(y_train, train_labels)
    x_test = np.concatenate((x_test, test_data), axis=0)
    y_test = np.append(y_test, test_labels)

train_data = None
train_labels = None
test_data = None
test_labels = None
    
# randomize the dataset 
permutation = np.random.permutation(y_train.shape[0])
x_train = x_train[permutation, :]
y_train = y_train[permutation]
    
permutation = np.random.permutation(y_test.shape[0])
x_test = x_test[permutation, :]
y_test = y_test[permutation]

np.save("/amax/data/hys/other/quickdraw/npy/all/x_train.npy", x_train)
np.save("/amax/data/hys/other/quickdraw/npy/all/y_train.npy", y_train)
np.save("/amax/data/hys/other/quickdraw/npy/all/x_test.npy", x_test)
np.save("/amax/data/hys/other/quickdraw/npy/all/y_test.npy", y_test)