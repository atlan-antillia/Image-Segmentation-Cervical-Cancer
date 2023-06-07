#
# ImageTileSplitter.py
# Spitting an image to multipe tiled images.
# 2023/06/06 to-arai

import os
import sys
import glob
import shutil
from PIL import Image
import traceback

class ImageTilesSplitter:

  def __init__(self, debug= False, split_size=256):
    self.split_size = split_size
    self.debug      = debug


  def split(self, image_files_pattern, output_dir):
    image_files = glob.glob(image_files_pattern)
    if len(image_files) == 0:
      raise Exception("Not found image files in {}".format(image_files_pattern))
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    print("--- all image files {}".format(image_files ))
    #input("HIT any key")
    for image_file in image_files:
      self.split_one(image_file, output_dir)


  def split_one(self, image_file, output_dir):
    if not os.path.exists(image_file):
      raise Exception("Not found image_file {}".format(image_file))
  
    image = Image.open(image_file)
    print("---Image2TilesSplitter split_one image_file {}".format(image_file))
    image = image.convert("RGB")
    w, h = image.size
    print(" w {} h {}".format(w, h))
    horiz_split_num = int(w/self.split_size) 
    vert_split_num  = int(h/self.split_size) 
    print("horiz_split_num {}".format(horiz_split_num))
    print("vert_split_num  {}".format(vert_split_num))
    layout =  str(horiz_split_num) + "&" + str(vert_split_num) 
    if horiz_split_num == 0:
      horiz_split_num = 1

    if vert_split_num == 0:
      vert_split_num = 1
  
    t = 1000
    tiles = []
    for j in range(vert_split_num):
      tile  = {}
      for i in range(horiz_split_num):
        t += 1
        #(left, upper, right, lower)
        left  = self.split_size * i
        upper = self.split_size * j
        right = left  + self.split_size
        lower = upper + self.split_size
 
        cropped_image = image.crop((left, upper, right, lower))
        basename    = os.path.basename(image_file)
        nameonly    = basename.split(".")[0]
        row_column  = "_(" + str(i) + ", " + str(j) + ")_"
        cropped_image_file = nameonly  + str(t) + row_column  + "#" + str(left) + "x" + str(upper) + "#" + ".png"
        tile["x_y"]   = (left, upper)
        tile["i_j"]   = (i, j)
        tile["image"] = cropped_image
        tile["mask"]  = None
        
        tiles.append(tile)

        output_image_file = os.path.join(output_dir, cropped_image_file)
        if self.debug:
          print("output filename {}".format(output_image_file))
          #input("HIT")
          cropped_image.save(output_image_file, "PNG")
    return tiles


if __name__ == "__main__":

  image_files_pattern = "./*.jpg"
  output_dir          = "./new_tiled_images/"
  try:
    splitter = ImageTilesSplitter()
    splitter.split(image_files_pattern, output_dir)

  except:
    traceback.print_exc()



    