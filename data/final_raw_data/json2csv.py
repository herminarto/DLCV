# To run this script, type command below to your terminal!
# ~$ python3 json2csv.py

import json
import numpy as np
import cv2
import operator
from PIL import Image 
from progress.bar import ShadyBar
from operator import itemgetter

with open("data.json", "r") as jdata:
    data = json.load(jdata)
    jdata.close()

def xStart(i):
    arr = []
    for j in range(0, 4):
        arr.append(data[i]["Label"]["Fire"][0][j]["x"])
    
    return min(arr)

def yStart(i):
    arr = []
    for j in range (0, 4):
        arr.append(data[i]["Label"]["Fire"][0][j]["y"])

    # yStart = height of images - y max
    return hn - max(arr)

def fireWidth(i):
    arr = []
    for j in range(0, 4):
        arr.append(data[i]["Label"]["Fire"][0][j]["x"])
    
    return max(arr) - min(arr)

def fireHeight(i):
    arr = []
    for j in range (0, 4):
        arr.append(data[i]["Label"]["Fire"][0][j]["y"])

    return max(arr) - min(arr)

def drawFireLocation(fname, x, y, w, h):
    if x is None:
        x, y, w, h = 0, 0, 0, 0
    
    # convert filename of images from int to str
    fname = str(fname).zfill(5) + ".jpg"

    # path1 = "path/to/input/directory/of/images/ + filename
    path1 = "/home/sahrul/Documents/project1/final/img/" + fname
    
    # read and draw rectangle of fire object to images
    img = cv2.imread(path1, cv2.IMREAD_COLOR)
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
    
    # path2 = "path/to/output/directory/of/images/ + filename
    path2 = "/home/sahrul/Documents/project1/final/img/loc/" + fname
    
    # create and save new file of images
    cv2.imwrite(path2, img)

arr1 = [] # non-normalized data
arr2 = [] # normalized data

a = 0 # number of normal images
b = 0 # number of fire images

bar = ShadyBar('Processing', max = len(data))

for i in range(0, len(data)):
    # n = filename of images
    n = str(data[i]["External ID"])

    # m = model of images, 1 = fire, 0 = no-fire
    m = int(data[i]["Label"]["model"])

    # src = "/path/to/src/directory/" + n = filename of images
    src = "/home/sahrul/Documents/project1/final/img/" + n
    
    # wn = width of images, hn = height of images
    wn, hn = Image.open(src).size

    # slicing str of images filename and convert to int
    n = int(n[:5])

    if m == 0:
        # x = xStart, y = yStart, w = fireWidth, h = fireHeight
        x, y, w, h = None, None, None, None

        # append to list as normalized label
        arr2.append([n, m, x, y, w, h])
        
        a += 1

    elif m == 1:
        x = xStart(i)
        y = yStart(i)       
        w = fireWidth(i)
        h = fireHeight(i)

        arr2.append([
            n,
            m,
            round(x/wn, 3),
            round(y/hn, 3),
            round(w/wn, 3),
            round(h/hn, 3)
        ])
        
        b += 1

    # un-comment line code below to mapping of fire object from images
    drawFireLocation(n, x, y, w, h)
    
    # append list as non-normalized label
    arr1.append([n, m, x, y, w, h])
    
    bar.next()

# convert list to numpy array
arr1, arr2 = np.array(arr1), np.array(arr2)

# sort multidimensional array by first column (idx = 0)
arr1, arr2 = arr1[np.argsort(arr1[:,0])], arr2[np.argsort(arr2[:,0])]

# create temporary lists
li1, li2 = [], []
for i in range(0, len(data)):
    # delete first element each array (idx = 0)
    temp1, temp2 = np.delete(arr1[i], 0), np.delete(arr2[i], 0)

    li1.append(temp1)
    li2.append(temp2)

# convert list to numpy array
arr1, arr2 = np.array(li1), np.array(li2)

# transpose array or matrix
arr1, arr2 = np.transpose(arr1), np.transpose(arr2)

# file1 = "path/to/file/directory/file.csv"
file1 = "label/non-normalized_label.csv"
with open(file1, "w") as fname:
    np.savetxt(fname, arr1, fmt = "%s", delimiter = ",")

# file2 = "path/to/file/directory/file.csv"
file2 = "label/normalized_label.csv"
with open(file2, "w") as fname:
    np.savetxt(fname, arr2, fmt = "%s", delimiter = ",")

bar.finish()

print ("---------------------------------------------------------")
print ("model [1]: {0} images\nmodel [0]: {1} images".format(b, a))
print ("---------------------------------------------------------")
print ("{} has been created.".format(file1))
print ("{} has been created.".format(file2))
print ("---------------------------------------------------------")
print ("process has been completed!")
print ("---------------------------------------------------------")