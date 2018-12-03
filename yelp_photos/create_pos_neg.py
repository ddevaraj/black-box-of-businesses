import json
import argparse
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-p","--dataset",required=True, help="Path to the input image")
ap.add_argument("-f1","--file1",required=True, help="Path to the json file 1")
ap.add_argument("-f2","--file2",required=True, help="Path to the json file 2")
args = vars(ap.parse_args())

directory_name = args["dataset"]
total_count = 0

negative_photo_ids = []

def create_list_of_negative_photo_ids(file_name):
        with open(file_name) as fp:
            line = fp.readline()
            while line:
                negative_photo_ids.append(line.strip())
                line = fp.readline()
            return len(negative_photo_ids)

def create_postxt(abs_dirname):
    """Move files into subdirectories."""
	
    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]
    i = 0   
    fp = open("positive.txt","w")
    fn = open("negative.txt","w")
    for f in files:
        # move file to current dir
        f_base = os.path.basename(f)
        f_base1 = f_base.split('.')	
	
        if f_base1[0] in negative_photo_ids:
            fn.write(f_base1[0])
            fn.write("\n")
        else:
            fp.write(f_base1[0])
            fp.write("\n")
    fp.close()


count1 = 0
count2 = 0
count1+=create_list_of_negative_photo_ids(args["file1"])
count2+=create_list_of_negative_photo_ids(args["file2"])
count2-=count1
total_count = count1 + count2
print("No of images in file 1 are : "+str(count1))
print("No of images in file 2 are : "+str(count2))
print("Total no of images are : "+str(total_count))
print("\n[INFO]Cretating postive txt\n")
create_postxt(directory_name)
