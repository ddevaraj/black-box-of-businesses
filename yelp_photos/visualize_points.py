from PIL import Image
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import sys
import argparse
from imutils import paths
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-d1","--dataset1",required=True, help="Path to the input image")
ap.add_argument("-d2","--dataset2",required=True, help="Path to the input image")
args = vars(ap.parse_args())

directory_name1 = args["dataset1"]
directory_name2 = args["dataset2"]
positive_points = []
negative_points = []
max_negative = 0
max_positive = 0
min_positive = 50000

a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
g = 0
h = 0
k = 0
j = 0

for i in os.listdir(directory_name1):
	s = directory_name1 + '/' + i
	im = Image.open(s)
        width,height = im.size
        resolution = width*height
        positive_points.append(resolution)
        if min_positive > resolution:
        	min_positive = resolution
        if max_positive< resolution:
                max_positive = resolution        
	size = resolution

        if size < 10000:
                a+=1
        elif size < 30000:
                b+=1
        elif size < 50000:
                c+=1
        elif size < 75000:
                d+=1
        elif size < 100000:
                e+=1
        elif size < 125000:
                f+=1
        elif size < 150000:
                g+=1
        elif size < 175000:
                h+=1
        elif size < 200000:
                k+=1
        else:
                j+=1

total_positive = float(a+b+c+d+e+f+g+h+k+j)
print("Total postive are :"+str(total_positive))

print("Min of positive points is :"+str(min_positive))	
print("Max of positive points is :"+str(max_positive))
print("")
print("The file size statistics for positive are as follows")
print("")
print("No of images in range 0      - 10000 are : "+str((a/total_positive)*100)+"%")
print("No of images in range 10000  - 30000 are : "+str((b/total_positive)*100)+"%")
print("No of images in range 30000  - 50000 are : "+str((c/total_positive)*100)+"%")
print("No of images in range 50000  - 75000 are : "+str((d/total_positive)*100)+"%")
print("No of images in range 75000  - 100000 are : "+str((e/total_positive)*100)+"%")
print("No of images in range 100000 - 125000 are : "+str((f/total_positive)*100)+"%")
print("No of images in range 125000 - 150000 are : "+str((g/total_positive)*100)+"%")
print("No of images in range 150000 - 175000 are : "+str((h/total_positive)*100)+"%")
print("No of images in range 175000 - 200000 are : "+str((k/total_positive)*100)+"%")
print("No of images in range > 200000 are : "+str((j/total_positive)*100)+"%")
print("")

a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
g = 0
h = 0
k = 0
j = 0


for i in os.listdir(directory_name2):
        s = directory_name2 + '/' + i
        im = Image.open(s)
        width,height = im.size
        resolution = width*height
        negative_points.append(resolution)
        if max_negative < resolution:
		max_negative = resolution
        size = resolution

        if size < 10000:
                a+=1
        elif size < 30000:
                b+=1
        elif size < 50000:
                c+=1
        elif size < 75000:
                d+=1
        elif size < 100000:
                e+=1
        elif size < 125000:
                f+=1
        elif size < 150000:
                g+=1
        elif size < 175000:
                h+=1
        elif size < 200000:
                k+=1
        else:
                j+=1
print("Max of negative points is :"+str(max_negative))
total_negative = float(a+b+c+d+e+f+g+h+k+j)
print("Total negative are :"+str(total_negative))
print("")
print("The file size statistics for negative are as follows")
print("")
print("No of images in range 0      - 10000 are : "+str((a/total_negative)*100)+"%")
print("No of images in range 10000  - 30000 are : "+str((b/total_negative)*100)+"%")
print("No of images in range 30000  - 50000 are : "+str((c/total_negative)*100)+"%")
print("No of images in range 50000  - 75000 are : "+str((d/total_negative)*100)+"%")
print("No of images in range 75000  - 100000 are : "+str((e/total_negative)*100)+"%")
print("No of images in range 100000 - 125000 are : "+str((f/total_negative)*100)+"%")
print("No of images in range 125000 - 150000 are : "+str((g/total_negative)*100)+"%")
print("No of images in range 150000 - 175000 are : "+str((h/total_negative)*100)+"%")
print("No of images in range 175000 - 200000 are : "+str((k/total_negative)*100)+"%")
print("No of images in range > 200000 are : "+str((j/total_negative)*100)+"%")
print("")

val = 0

plt.style.use("ggplot")
plt.figure()
plt.plot(positive_points, np.zeros_like(positive_points)+val, label="positive")
plt.plot(negative_points, np.zeros_like(negative_points)+1, label="negative")
plt.title("Positive Points and Negative Points")
plt.legend()
plt.savefig("postive_and_negative_points.png")
plt.close()

plt.hist(positive_points,bins=30)
plt.xlabel('Resolution')
plt.ylabel('# No of images')
plt.title("Positive Points")
plt.savefig("positive_points_hist.png")
plt.close()

plt.hist(negative_points,bins=30)
plt.xlabel('Resolution')
plt.ylabel('# No of images')
plt.title("Negative Points")
plt.savefig("negative_points_hist.png")
plt.close()

plt.hist(positive_points,bins=30)
plt.hist(negative_points,bins=30)
plt.xlabel('Resolution')
plt.ylabel('# No of images')
plt.title("Positive and Negative Points")
plt.legend(loc='upper left')
plt.savefig("positive_and_negative_points_hist.png")
plt.close()

'''
plt.style.use("ggplot")
plt.figure()
plt.plot(negative_points, label="negative")
plt.title("Negative Points")
plt.legend()
plt.savefig("negative_points.png")
'''
