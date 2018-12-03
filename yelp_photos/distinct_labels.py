import json
  
data = []

with open('../yelp_academic_dataset_photo.json') as f:
        for line in f:
                data.append(json.loads(line))

labels = {}
for i in data:
        if labels.get(i["label"])!=None:
                y = labels.get(i["label"])
                labels[i["label"]] = y + 1
        else:
                labels[i["label"]] = 1

print("The distinct labels are as follows")
print(labels)
