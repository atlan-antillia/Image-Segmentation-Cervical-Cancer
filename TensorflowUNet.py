# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This is based on the code in the following web sites:

# 1. Keras U-Net starter - LB 0.277
# https://www.kaggle.com/code/keegil/keras-u-net-starter-lb-0-277/notebook

# 2. U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/pdf/1505.04597.pdf

# You can customize your TensorflowUnNet model by using a configration file
# Example: train.config

# 2023/05/28 Added dilation_rate parameter to Conv2D
# 2023/05/28 Modified to read loss and metrics from train_eval_infer.config file.

"""
[model]
image_width    = 256
image_height   = 256
image_channels = 3

num_classes    = 1
base_filters   = 16
num_layers     = 8
dropout_rate   = 0.05
learning_rate  = 0.001
"""

# 2023/06/07 Added 
  # 1 Split the orginal image to some tiled-images
  # 2 Infer segmentation regions on those images 
  # 3 Merge detected regions into one image
#
#  def infer_tiles(self, input_dir, output_dir, expand=True):
   


import os
import random

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import shutil
import sys
import glob
import traceback
import numpy as np
import cv2
import tensorflow as tf
import random

from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D, Dropout, Conv2D, MaxPool2D, BatchNormalization

from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.activations import elu, relu
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
from losses import iou_coef, iou_loss, bce_iou_loss

from ConfigParser import ConfigParser

from EpochChangeCallback import EpochChangeCallback
from GrayScaleImageWriter import GrayScaleImageWriter
from PIL import Image

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"
TILEDINFER = "tiledinfer"

BEST_MODEL_FILE = "best_model.h5"

class TensorflowUNet:

  def __init__(self, config_file):
    self.set_seed()

    self.config    = ConfigParser(config_file)
    image_height   = self.config.get(MODEL, "image_height")

    image_width    = self.config.get(MODEL, "image_width")
    image_channels = self.config.get(MODEL, "image_channels")

    num_classes    = self.config.get(MODEL, "num_classes")
    base_filters   = self.config.get(MODEL, "base_filters")
    num_layers     = self.config.get(MODEL, "num_layers")
 

    if not (image_width == image_height and  image_width % 128 == 0 and image_height % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    self.model     = self.create(num_classes, image_height, image_width, image_channels, 
                            base_filters = base_filters, num_layers = num_layers)
    
    learning_rate  = self.config.get(MODEL, "learning_rate")

    self.optimizer = Adam(learning_rate = learning_rate, 
         beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, 
         amsgrad=False) 

    #  Modified to read loss and metrics from train_eval_infer.config file.
    binary_crossentropy = tf.keras.metrics.binary_crossentropy
    binary_accuracy     = tf.keras.metrics.binary_accuracy

    # Default loss and metrics functions
    self.loss    = binary_crossentropy
    self.metrics = [binary_accuracy]
    
    # Read a loss function name from our config file, and eval it.
    # loss = "binary_crossentropy"
    loss = self.config.get(MODEL, "loss")
    print("=== loss {}".format(loss))
    self.loss  = eval(loss)

    # Read a list of metrics function names, ant eval each of the list,
    # metrics = ["binary_accuracy"]
    metrics  = self.config.get(MODEL, "metrics")
    self.metrics = []
    for metric in metrics:
      self.metrics.append(eval(metric))
    
    print("--- loss    {}".format(self.loss))
    print("--- metrics {}".format(self.metrics))

    self.model.compile(optimizer = self.optimizer, loss= self.loss, metrics = self.metrics)
    show_summary = self.config.get(MODEL, "show_summary")
    if show_summary:
      self.model.summary()    
    self.model_loaded = False


  def set_seed(self, seed=137):
    print("=== set seed {}".format(seed))
    random.seed    = seed
    np.random.seed = seed
    tf.random.set_seed(seed)

  def create(self, num_classes, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
    # inputs
    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    inputs = Input((image_height, image_width, image_channels))
    s= Lambda(lambda x: x / 255)(inputs)

    # Encoder
    dropout_rate = self.config.get(MODEL, "dropout_rate")
    enc         = []

    pool_size    = (2, 2)
    #kernel_sizes = [(7,7), (5,5)]
    # <experiment on="2023/06/07"> 
    kernel_sizes = [(5, 5), (3,3)]
    for n in range(num_layers-2):
      kernel_sizes  += [(3,3)]  
    rkernel_sizes =  kernel_sizes[::-1]
    print("--- kernel_size   {}".format(kernel_sizes))
    print("--- rkernel_size  {}".format(rkernel_sizes))
    # </experiment>

    dilation    = self.config.get(MODEL, "dilation")
    print("=== dilation {}".format(dilation))      
    #kernel_sizes = self.config.get(MODEL, "kernel_sizes")
  
    strides = (1,1)
    for i in range(num_layers):
      filters = base_filters * (2**i)
      kernel_size = kernel_sizes[i] #random.choice(kernel_sizes)
    
  
      c = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(s)
      c = Dropout(dropout_rate * i)(c)
      c = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(c)
      # 2023/06/06 Added the following block
      #c = Dropout(dropout_rate * i)(c)
      #c = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
      #           kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(c)
      #c = BatchNormalization(c)
    
      if i < (num_layers-1):
        p = MaxPool2D(pool_size=pool_size)(c)
        s = p
      enc.append(c)

    #print(enc)
    enc_len = len(enc)
    print("---len {}".format(enc_len))
    enc.reverse()
    n = 0
    c = enc[n]
    
    # --- Decoder
   
    for i in range(num_layers-1):
      kernel_size = rkernel_sizes[i] #random.choice(kernel_sizes)

      f = enc_len - 2 - i
      filters = base_filters* (2**f)
      #for kernel_size in reversed(kernel_sizes):
      u = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c)
      n += 1
      u = concatenate([u, enc[n]])
      u = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(u)
      u = Dropout(dropout_rate * f)(u)
      u = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(u)
      
      # 2023/06/06 Added the following block      
      #u = Dropout(dropout_rate * f)(u)
      #u = Conv2D(filters, kernel_size, strides=strides, activation=relu, 
      #           kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(u)
      #u = BatchNormalization(u)
      
      c  = u

    # outouts
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c)

    # create Model
    model = Model(inputs=[inputs], outputs=[outputs])

    return model


  def train(self, x_train, y_train): 
    batch_size = self.config.get(TRAIN, "batch_size")
    epochs     = self.config.get(TRAIN, "epochs")
    patience   = self.config.get(TRAIN, "patience")
    eval_dir   = self.config.get(TRAIN, "eval_dir")
    model_dir  = self.config.get(TRAIN, "model_dir")
    metrics    = ["accuracy", "val_accuracy"]
    try:
      metrics    = self.config.get(TRAIN, "metrics")
    except:
      pass
    if os.path.exists(model_dir):
      shutil.rmtree(model_dir)

    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    weight_filepath   = os.path.join(model_dir, BEST_MODEL_FILE)

    early_stopping = EarlyStopping(patience=patience, verbose=1)
    check_point    = ModelCheckpoint(weight_filepath, verbose=1, save_best_only=True)
    epoch_change   = EpochChangeCallback(eval_dir, metrics)

    history = self.model.fit(x_train, y_train, 
                    validation_split=0.2, batch_size=batch_size, epochs=epochs, 
                    shuffle=False,
                    callbacks=[early_stopping, check_point, epoch_change],
                    verbose=1)

  def load_model(self) :
    rc = False
    if  not self.model_loaded:    
      model_dir  = self.config.get(TRAIN, "model_dir")
      weight_filepath = os.path.join(model_dir, BEST_MODEL_FILE)
      if os.path.exists(weight_filepath):
        self.model.load_weights(weight_filepath)
        self.model_loaded = True
        print("=== Loaded a weight_file {}".format(weight_filepath))
        rc = True
      else:
        message = "Not found a weight_file " + weight_filepath
        raise Exception(message)
    else:
      print("== Already loaded a weight file loaded ")
    return rc


  # 2023/05/05 Added newly.    
  def infer(self, input_dir, output_dir, expand=True):
    writer       = GrayScaleImageWriter()
    
    image_files  = glob.glob(input_dir + "/*.png")
    image_files += glob.glob(input_dir + "/*.jpg")
    image_files += glob.glob(input_dir + "/*.tif")
    image_files += glob.glob(input_dir + "/*.bmp")

    width        = self.config.get(MODEL, "image_width")
    height       = self.config.get(MODEL, "image_height")
    # 2023/05/24
    merged_dir   = None
    try:
      merged_dir = self.config.get(INFER, "merged_dir")
      if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
      if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    except:
      pass

    for image_file in image_files:
      basename = os.path.basename(image_file)
      name     = basename.split(".")[0]
      img      = cv2.imread(image_file, cv2.COLOR_BGR2RGB)
      h = img.shape[0]
      w = img.shape[1]
      # Any way, we have to resize input image to match the input size of our TensorflowUNet model.
      img         = cv2.resize(img, (width, height))
      predictions = self.predict([img], expand=expand)
      prediction  = predictions[0]
      image       = prediction[0]    
      # Resize the predicted image to be the original image size (w, h), and save it as a grayscale image.
      # Probably, this is a natural way for all humans. 
      mask = writer.save_resized(image, (w, h), output_dir, name)
      # 2023/05/24
      print("--- image_file {}".format(image_file))
      if merged_dir !=None:
        # Resize img to the original size (w, h)
        img   = cv2.resize(img, (w, h))
        print("=== mask {}".format(mask.shape))
        print("=== image {}".format(img.shape))

        img += mask
        merged_file = os.path.join(merged_dir, basename)
        cv2.imwrite(merged_file, img)     

  def predict(self, images, expand=True):
    self.load_model()

    predictions = []
    for image in images:
      #print("=== Input image shape {}".format(image.shape))
      if expand:
        image = np.expand_dims(image, 0)
      pred = self.model.predict(image)
      predictions.append(pred)
    return predictions    


  # 2023/06/05
  # 1 Split the orginal image to some tiled-images
  # 2 Infer segmentation regions on those images 
  # 3 Merge detected regions into one image

  def infer_tiles(self, input_dir, output_dir, expand=True):
    writer       = GrayScaleImageWriter()
    
    image_files  = glob.glob(input_dir + "/*.png")
    image_files += glob.glob(input_dir + "/*.jpg")
    image_files += glob.glob(input_dir + "/*.tif")
    image_files += glob.glob(input_dir + "/*.bmp")

    merged_dir   = None
    try:
      merged_dir = self.config.get(TILEDINFER, "merged_dir")
      if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
      if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    except:
      pass

    for image_file in image_files:
      image = Image.open(image_file)
      w, h  = image.size
      split_size  =  self.config.get(MODEL, "image_width")

      # 2023/06/10 w <-> h
      vert_split_num  = h // split_size + 1
      horiz_split_num = w // split_size + 1
      background      = Image.new("L", (w, h))

      for j in range(vert_split_num):
        for i in range(horiz_split_num):
          left  = split_size * i
          upper = split_size * j
          right = left  + split_size
          lower = upper + split_size
 
          cropped = image.crop((left, upper, right, lower))  
    
          predictions = self.predict([cropped], expand=expand)
          prediction  = predictions[0]
          mask         = prediction[0]    
          img          = self.mask_to_image(mask)
          background.paste(img, (left, upper))
          print("---paste j:{} i:{}".format(j, i))
          #input("HHHIT")  
      basename = os.path.basename(image_file)
      output_file = os.path.join(output_dir, basename)
      #background.show()
      #input("----")
      background.save(output_file)
      
      if merged_dir !=None:
        # Resize img to the original size (w, h)
        img   = np.array(image)
        img   = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask  = np.array(background)
        mask   = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img += mask

        merged_file = os.path.join(merged_dir, basename)
        cv2.imwrite(merged_file, img)     

  def mask_to_image(self, data, factor=255.0):
    
    h = data.shape[0]
    w = data.shape[1]

    data = data*factor
    data = data.reshape([w, h])
    image = Image.fromarray(data)
    return image
    """
    image = Image.new("L", (w, h))
    
    for i in range(w):
      for j in range(h):
        z = data[j][i]
        if type(z) == list:
          z = z[0]
        v = int(z * factor)
        image.putpixel((i,j), v)
 
    return image
    """


  def evaluate(self, x_test, y_test): 
    self.load_model()

    score = self.model.evaluate(x_test, y_test, verbose=1)
    #print("score {}".format(score))
     
    
if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    
    # Create a UNetMolde and compile
    model    = TensorflowUNet(config_file)
    
    """
    datatset = ImageMaskDataset(config_file)
    x_train, y_train  = dataset.create(dataset=TRAIN)

    model.train(x_train, y_train)
    """

  except:
    traceback.print_exc()
    
