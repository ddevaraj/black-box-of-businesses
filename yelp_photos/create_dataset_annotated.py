import json
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p","--dataset",required=True, help="Path to the input image")
ap.add_argument("-f1","--file1",required=True, help="Path to the json file 1")
ap.add_argument("-f2","--file2",required=True, help="Path to the json file 2")
args = vars(ap.parse_args())

directory_name = args["dataset"]
total_count = 0

negative_photo_ids = []

def negative_photo(file_name):
        count = 0
        with open(file_name) as fp:
                line = fp.readline() 
                length = len(line.split(","))
                i = 1
                for x in line.split(","):
                        if i==1:
                                print(x[1:])
                        elif i==length:
                                print(x[1:len(x)-1])
                        else:
                                print(x[1:])
                        i+=1
                        count+=1
        print("Length is : "+str(length))
        return count

count1 = 0
count2 = 0
count1+=negative_photo(args["file1"])
count2+=negative_photo(args["file2"])
total_count = count1 + count2
print("No of images in file 1 are : "+str(count1))
print("No of images in file 2 are : "+str(count2))
print("Total no of images are : "+str(total_count))
