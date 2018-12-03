import argparse
import csv
import os

ap = argparse.ArgumentParser()
ap.add_argument("-f","--file_name",required=True, help="Path to the input image")
ap.add_argument("-d","--dataset",required=True, help="Path to dataset")
ap.add_argument("-o","--output",required=True, help="Path to output")
args = vars(ap.parse_args())

file        = args["file_name"]
dataset     = args["dataset"]
out_file = args["output"]

input_file  = csv.reader(open(file,'r'))
output_file = csv.writer(open(out_file,'w')) 

photo_ids = []

print("[INFO] Creating photo_id dictionary")

for i in os.listdir(dataset):
        s = i.split(".")
        photo_ids.append(s[0])

i = 0
no_of_rows_in_output_file = 0

print("Size of photo_ids is : "+str(len(photo_ids)))

print("[INFO] Started writing to output file")

for row in input_file:
	if i==0:
		i+=1
		output_file.writerow(row)
		continue
	if row[0] in photo_ids:
		output_file.writerow(row)
		no_of_rows_in_output_file+=1
	i+=1

print("No of row in input file is : "+str(i))
print("No of rows in output file is : "+str(no_of_rows_in_output_file))
