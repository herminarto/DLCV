# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:31:57 2018

@author: herminarto.nugroho
"""

# To run this script, type command below to your terminal!
# ~$ python3 resizeImages.py /path/to/input/directory/ size /path/to/ouput/directory/

import os
import sys
from PIL import Image

def readf():
    path  = str(sys.argv[1].rstrip('/')) # path to input directory (no file directory)
    size   = str(sys.argv[2]) # images size in pixel
    out  = str(sys.argv[3].rstrip('/')) # path to output directory
    
    tclass = [ d for d in os.listdir(path) ]
    
    if not os.path.exists(out):
        os.makedirs(out)
    
    for x in tclass:
        list1 = os.path.join(path + '/'+ x)
        list2 = os.path.join(out + '/'+ x)
        img = Image.open(list1)
        img = img.resize((int(size), int(size)), Image.ANTIALIAS)
        img.save(list2, "JPEG", quality = 100)
        print ("[STATUS] - resizing file : {0}".format(x))

readf()