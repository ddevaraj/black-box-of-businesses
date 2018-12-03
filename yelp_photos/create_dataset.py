import os
import argparse
from imutils import paths
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-p","--dataset",required=True, help="Path to the input image")
args = vars(ap.parse_args())

directory_name = args["dataset"]
count = 0

def create_photo_id_to_review_dictionary():
        with open('sentiment_reviews.txt') as fp:
                line = fp.readline()
                while line:
                        line_val = line.split("\t")
                        photo_id_to_review[line_val[0]] = int(line_val[1])
                        line = fp.readline()

def move_files(abs_dirname):
    """Move files into subdirectories."""
	
    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]
    print(len(files))
    i = 0   
    
    positive_subdir = '/Users/sai/Documents/Data_Mining_INF_553/Project/yelp_photos/code/Dataset/Positive'
    negative_subdir = '/Users/sai/Documents/Data_Mining_INF_553/Project/yelp_photos/code/Dataset/Negative'
    neutral_subdir = '/Users/sai/Documents/Data_Mining_INF_553/Project/yelp_photos/code/Dataset/Neutral'
    
    for f in files:
        # move file to current dir
        f_base = os.path.basename(f)
        f_base1 = f_base.split('.')
	
	
        if photo_id_to_review[f_base1[0]]==1:
		shutil.copy(f, positive_subdir)
	elif photo_id_to_review[f_base1[0]]==-1:
                shutil.copy(f, negative_subdir)
        else:
                shutil.copy(f, neutral_subdir)
	
photo_id_to_review = {}
create_photo_id_to_review_dictionary()
move_files(directory_name)

print("Length of dictionary is :"+str(len(photo_id_to_review)))
print("No of images in the dataset are :"+str(count))
