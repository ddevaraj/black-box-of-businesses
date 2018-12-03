from PIL import Image
import os
import sys
import argparse
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-d1","--dataset1",required=True, help="Path to the input image")
ap.add_argument("-d2","--dataset2",required=True, help="Path to the input image")
args = vars(ap.parse_args())

directory_name1 = args["dataset1"]
directory_name2 = args["dataset2"]

resolutions = open("resolutions.txt","w")
resolutions_to_id = open("id_to_resolutions.txt","w")

for i in os.listdir(directory_name1):
	s = directory_name1 + '/' + i
	im = Image.open(s)
        width,height = im.size
        resolution = width*height
        resolutions.write(str(resolution)+'\n')
	resolutions_to_id.write(i+" "+str(resolution)+'\n')

for i in os.listdir(directory_name2):
        s = directory_name2 + '/' + i
        im = Image.open(s)
        width,height = im.size
        resolution = width*height
        resolutions.write(str(resolution)+'\n')
	resolutions_to_id.write(i+" "+str(resolution)+'\n')

