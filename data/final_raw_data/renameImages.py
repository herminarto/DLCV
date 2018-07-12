# To run this script, type command below to your terminal!
# ~$ python3 renameImages.py

import os
import random

i = 1 # start index of images name

# path_src = /path/to/input/directory/
path_src = "/home/sahrul/Documents/project1/final/img/"

for filename in os.listdir(path_src):
    dst = str(i).zfill(5) + ".jpg"
    src = path_src + filename
    
    # dst = /path/to/ouput/directory/
    dst = '/home/sahrul/Documents/project1/final/temp/' + dst

    os.rename(src, dst)

    print ("[STATUS] - {0} has been change to {1}".format(src, dst))
    
    i += 1