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
# ImageMaskDatsetGenerator.py

# 2023/06/07 Toshiyuki Arai @antillia.com


import os
import numpy as np
import shutil
import glob
import cv2
import traceback

#from PIL import Image, ImageDraw, ImageFilter

class ImageMaskDatasetGenerator:

  def __init__(self, data_dir="./", output_dir="./master", width=256, height=256, 
               rotation=False, debug=False):
   
    self.data_dir   = data_dir
    self.output_dir = output_dir
    self.W          = width
    self.H          = height
    self.debug      = debug
    self.rotation   = rotation
    self.IMAGES_DIR = "images"
    self.MASKS_DIR  = "masks"
    self.output_img = ".jpg"
    self.TRAIN      = "train"
    self.VALID      = "valid"
    self.TEST       = "test"
    self.image_dirs = [
       "im_Dyskeratotic",
       "im_Koilocytotic",
       "im_Metaplastic",
       "im_Parabasal",
     ]
    
  def create_master(self, debug=False):

    for image_dir in self.image_dirs:
      # We use fullsize image (bmp) files not CROPPED
      # image_files = glob.glob(root_dir + "/" + image_dir + "/" + image_dir + "/CROPPED/*bmp")
      
      image_files = glob.glob(root_dir + "/" + image_dir + "/" + image_dir + "/*bmp")
      # Get category name from images_dir string array.
      category   = image_dir.split("_")[1]
      
      num_images = len(image_files)
      num_train  = int(num_images * 0.7)
      num_valid  = int(num_images * 0.2)
      num_test   = int(num_images * 0.1)
      train_image_files = image_files[0: num_train]
      valid_image_files = image_files[num_train: num_train + num_valid]
      test_image_files  = image_files[num_train + num_valid:]
      print("num_train {}".format(num_train))
      print("num_valid {}".format(num_valid))
      print("num_test {}".format(num_test))
      
      self.create_dataset(train_image_files, image_dir, output_dir, self.TRAIN, category, debug=debug)
      self.create_dataset(valid_image_files, image_dir, output_dir, self.VALID, category, debug=debug)
      self.create_dataset(test_image_files,  image_dir, output_dir, self.TEST,  category, debug=debug)

  def create_dataset(self, image_files, image_dir, output_dir, dataset, category, debug=False):
    output_dir = os.path.join(output_dir, dataset)
    categorized_output_dir = os.path.join(output_dir, category)

    if not os.path.exists(categorized_output_dir):
      os.makedirs(categorized_output_dir)
    
    output_images_dir = os.path.join(categorized_output_dir, "images")
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    output_masks_dir  = os.path.join(categorized_output_dir, "masks")
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)
        
    for image_file in image_files:
        dir_name = os.path.dirname(image_file)
        basename = os.path.basename(image_file)
        name     = basename.split(".")[0]
        
        data_files  = glob.glob(dir_name + "/" + name + "_cyt*.dat")

        image = cv2.imread(image_file)

        h, w,_ = image.shape

        category = image_dir.split("_")[1]
        if dataset != self.TEST:
          index = 1000
          for data_file in data_files:
            index += 1
            indexed_name = "_" + str(index) + "_" + name
            # When dataset is "train" or "valid", create a pair of image and mask files.
            self.save_image(dataset, output_images_dir, category, indexed_name, image)
            self.create_mask(dataset, output_masks_dir, w, h, data_file, category, indexed_name)
        else:
          self.save_image(dataset, output_images_dir, category, name, image)
          #When dataset == "test", create a merged mask file from data_files 
          self.create_merged_mask(dataset, output_masks_dir, w, h, data_files, category, name)


  def save_image(self, dataset, output_dir, category, name, image):
    resized = cv2.resize(image, dsize=(self.W, self.H))
    if self.rotation and dataset !=self.TEST:
      self.rotate(resized, category, name, output_dir)
      self.flip(resized, category, name, output_dir)
    else:
      resized_filename   =  category + "_" + name + self.output_img
      resized_filepath = os.path.join(output_dir, resized_filename )
      print("---- Saved {}".format(resized_filepath))
      cv2.imwrite(resized_filepath, resized)


  def create_mask(self, dataset, output_dir, w, h, data_file, category, name):
    categories = {
       "Dyskeratotic": (0,   255, 255),
       "Koilocytotic": (255, 255, 0),
       "Metaplastic":  (0,   255, 0),
       "Parabasal"  :  (255, 0,  128),
    }
    color  = categories [category]
    image  = np.zeros((h, w, 3), np.uint8 )
    points = self.get_mask_polygon(data_file)
    cv2.fillPoly(image, pts=[points],  color=color)
        
    resized = cv2.resize(image, (self.H, self.W))
    if self.rotation and dataset !=self.TEST:
      self.rotate(resized, category, name, output_dir)
      self.flip(resized, category, name, output_dir)
    else:
      resized_filename   =  category + "_" + name + self.output_img
      resized_filepath = os.path.join(output_dir, resized_filename )
      print("---- Saved {}".format(resized_filepath))
      cv2.imwrite(resized_filepath, resized)

   
  def create_merged_mask(self, dataset, output_dir, w, h, data_files, category, name):
    categories = {
       "Dyskeratotic": (0,   255, 255),
       "Koilocytotic": (255, 255, 0),
       "Metaplastic":  (0,   255, 0),
       "Parabasal"  :  (255, 0,  128), 
    }
    color   = categories [category]
    # Create an empty background image 
    image = np.zeros((h, w, 3), np.uint8 )
    #Merge all polygons defined in datafile onto image 
    for data_file in data_files:
      #print("=== data_file {}".format(data_file))
      points = self.get_mask_polygon(data_file)
      #print("=== polygon_points {}".format(points))
      # Draw a filled polygon on the image by a specified color depending the category
      cv2.fillPoly(image, pts=[points],  color=color)
        
    resized = cv2.resize(image, (self.H, self.W))

    if self.rotation and dataset !=self.TEST:
      self.rotate(resized, category, name, output_dir)
      self.flip(resized, category, name, output_dir)
    else:
      resized_filename   =  category + "_" + name + self.output_img
      resized_filepath = os.path.join(output_dir, resized_filename )
      print("---- Saved {}".format(resized_filepath))
      cv2.imwrite(resized_filepath, resized)


  def get_bounding_box(self, data_file):
    polygon = []
    with open(data_file, "r") as f:
      lines = f.readlines()
      for line in lines:
        line = line.replace("\n", "")
        ar = line.split(",")
        x = int( float(ar[0]) )
        y = int( float(ar[1]) )
        polygon.append([x, y])
      #print("--- polygon {}".format(polygon))

    points = np.array(polygon)
    x,y,w,h = cv2.boundingRect(points)
    return (x, y, w, h)


  def get_mask_polygon(self, data_file):
    polygon = []
    with open(data_file, "r") as f:
      lines = f.readlines()
      for line in lines:
        line = line.replace("\n", "")
        ar = line.split(",")
        x = int( float(ar[0]) )
        y = int( float(ar[1]) )
        polygon.append([x, y])
      #print("--- polygon {}".format(polygon))

    polygon_points = np.array(polygon)
    return   polygon_points 


  def rotate(self, image, category, name, output_dir):
    ANGLES = [0, 90, 180, 270]
    #ANGLES = [0, 60, 120, 180, 240, 280]
    print("=== Output_dir {}".format(output_dir))
    for angle in ANGLES:
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H))

      rotated_filename         = "rotated-" + str(angle) + "-" + "_" + category + "_" + name + self.output_img
      rotated_filepath = os.path.join(output_dir, rotated_filename )
      print("---- Saved {}".format(rotated_filepath))

      cv2.imwrite(rotated_filepath, rotated)


  def flip(self, image, category, name, output_dir):
    DIRECTIONS = [0, -1, 1]
    print("=== Output_dir {}".format(output_dir))
    for direction in DIRECTIONS:
      flipped = cv2.flip(image, direction)

      flipped_filename         = "flipped-" + str(direction) + "-" + "_" + category + "_" + name + self.output_img
      flipped_filepath = os.path.join(output_dir, flipped_filename )
      print("---- Saved {}".format(flipped_filepath))

      cv2.imwrite(flipped_filepath, flipped)

"""
Input
./
├─im_Dyskeratotic
│  └─im_Dyskeratotic
│      └─CROPPED
├─im_Koilocytotic
│  └─im_Koilocytotic
│      └─CROPPED
├─im_Metaplastic
│  └─im_Metaplastic
│      └─CROPPED
├─im_Parabasal
│  └─im_Parabasal
│      └─CROPPED
└─im_Superficial-Intermediate
    └─im_Superficial-Intermediate
        └─CROPPED 
"""


if __name__ == "__main__":
  try:
    root_dir   = "./"
    output_dir = "./256x256CervicalCancer"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    generator = ImageMaskDatasetGenerator(data_dir="./", output_dir=output_dir, rotation=True, debug=True)
    generator.create_master( debug=True)

  except:
    traceback.print_exc()

"""
Output
./256X256CervicalCancer
├─test
│  ├─Dyskeratotic
│  │  ├─images
│  │  └─masks
│  ├─Koilocytotic
│  │  ├─images
│  │  └─masks
│  ├─Metaplastic
│  │  ├─images
│  │  └─masks
│  └─Parabasal
│      ├─images
│      └─masks
├─train
│  ├─Dyskeratotic
│  │  ├─images
│  │  └─masks
│  ├─Koilocytotic
│  │  ├─images
│  │  └─masks
│  ├─Metaplastic
│  │  ├─images
│  │  └─masks
│  └─Parabasal
│      ├─images
│      └─masks
└─valid
    ├─Dyskeratotic
    │  ├─images
    │  └─masks
    ├─Koilocytotic
    │  ├─images
    │  └─masks
    ├─Metaplastic
    │  ├─images
    │  └─masks
    └─Parabasal
        ├─images
        └─masks
"""