# To run this script, type command below to your terminal!
# ~$ python3 downloadImages.py --urls urls.txt --output /path/to/your/output/directory

import os
import cv2
import requests
import argparse
from imutils import paths

ap = argparse.ArgumentParser()

# to parse input urls
ap.add_argument("-u", "--urls", required = True, help = None)

# to parse output direcotory
ap.add_argument("-o", "--output", required = True, help = None)
args = vars(ap.parse_args())
rows = open(args["urls"]).read().strip().split("\n")

counter = 1

for url in rows:
	try:
		r = requests.get(url, timeout = 60)

		# to set filename of images
		p = os.path.sep.join([args["output"], "{}.jpg".format(str(counter).zfill(5))])
		
		f = open(p, "wb")
		f.write(r.content)
		f.close()
		
		counter += 1

		print ("[STATUS] - downloaded: {0}".format(p))
	
	except:
		print ("[STATUS] - error downloading {} - skipping.".format(p))

for imagePath in paths.list_images(args["output"]):
	delete = False
	
	try:
		image = cv2.imread(imagePath)
		
		if image is None:
			delete = True

	except:
		print ("[STATUS] - Except")
		delete = True	
	
	if delete:
		print ("[STATUS] - deleting {}".format(imagePath))
		os.remove(imagePath)