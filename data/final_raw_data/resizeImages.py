# To run this script, type command below to your terminal!
# ~$ python3 resizeImages.py /path/to/input/directory/ size /path/to/ouput/directory/

import os
import sys
from PIL import Image

def readf():
    try:
        path1  = str(sys.argv[1].rstrip('/')) # path to input directory (no file directory)
        size   = str(sys.argv[2]) # images size in pixel
        path2  = str(sys.argv[3].rstrip('/')) # path to output directory
       
        print ("starting....")
        print ("colecting data from {} ".format(path1))
       
        tclass = [ d for d in os.listdir(path1) ]
        counter = 0
       
        for x in tclass:
            list1 = os.path.join(path1, x) # list of input directory
            list2 = os.path.join(path2 + '/', x + '/') # list of output direcotory
       
            if not os.path.exists(list2):
                os.makedirs(list2)
       
            if os.path.exists(list2):
                for d in os.listdir(list1):
                    try:
                        img = Image.open(os.path.join(path1 + '/' + x, d))
                        img = img.resize((int(size), int(size)), Image.ANTIALIAS)
                        
                        fname, extension = os.path.splitext(d)
                        
                        newfile = fname + extension
                        
                        if extension != ".jpg" :
                            newfile = fname + ".jpg"
                        
                        img.save(os.path.join(path2 + '/' + x, newfile), "JPEG", quality = 100)
                        
                        print ("[STATUS] - resizing file : {0} - {1}".format(x, d))
                    
                    except (Exception, e):
                        print ("[STATUS] - error resizing file : {0} - {1}".format(x, d))
                        
                        sys.exit(1)
               
                counter += 1
    
    except (Exception, e):
        print ("Error, check input directory etc : {}".format(e))
        
        sys.exit(1)

readf()