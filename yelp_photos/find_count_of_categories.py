import json

data = []

with open('../yelp_academic_dataset_photo.json') as f:
        for line in f:
                data.append(json.loads(line))

categories = {}
total_count = 0

for i in data:
        total_count+=1
	label = i["label"]
	if categories.get(label)!=None:
		x = categories.get(label)
		categories[label] = x + 1
	else:
                categories[label] = 1

print("Total count is : "+str(total_count))
print("Categories counts are as follows")
print(categories)
