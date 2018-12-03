import json

data = []
with open('../yelp_academic_dataset_photo.json') as f:
	for line in f:
		data.append(json.loads(line))
captions = {}

for i in data:
	if captions.get(i["caption"])!=None:
		x = captions.get(i["caption"])
		captions[i["caption"]] = x + 1
	else:
		captions[i["caption"]] = 1

print("The captions are as Follows")	
print(captions)
