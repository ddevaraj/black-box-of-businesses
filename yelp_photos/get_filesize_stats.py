import os
import sys
import argparse
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-p","--dataset",required=True, help="Path to the input image")
args = vars(ap.parse_args())

directory_name = args["dataset"]

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
min = sys.maxsize
max = 0

total_count = 0

for i in os.listdir(directory_name):
	total_count+=1
	s = directory_name + '/' + i
	size = os.path.getsize(s)
        
	if size < 5000:
		a+=1
	elif size < 10000:
		b+=1
	elif size < 15000:
		c+=1
	elif size < 20000:
		d+=1
	elif size < 25000:
		e+=1
	elif size < 30000:
		f+=1
        elif size < 34000:
                g+=1
        elif size < 40000:
                h+=1
        elif size < 45000:
                k+=1
        else:
                j+=1
        if size < min:
		min = size
        if size > max:
                max = size

print("")
print("The file size statistics are as follows")
print("")
print("No of images in range 0     - 5000 are : "+str(a))
print("No of images in range 5000 - 10000 are : "+str(b))
print("No of images in range 10000 - 15000 are : "+str(c))
print("No of images in range 15000 - 20000 are : "+str(d))
print("No of images in range 20000 - 25000 are : "+str(e))
print("No of images in range 25000 - 30000 are : "+str(f))
print("No of images in range 30000 - 34000 are : "+str(g))
print("No of images in range 35000 - 40000 are : "+str(h))
print("No of images in range 40000 - 45000 are : "+str(k))
print("No of images in range > 45000 are : "+str(j))
print("")
print("Min file size is :"+str(min))
print("Max file size is :"+str(max))

print("\nTotal no of files is : "+str(total_count))
