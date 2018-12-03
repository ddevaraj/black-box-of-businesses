import json
from  

data = []

with open('../yelp_academic_dataset_photo.json') as f:
        for line in f:
                data.append(json.loads(line))

sentiments = {}

for i in data:
	caption_text = i["caption"]
	sentiment = client.Sentiment({'text': caption_text})
	if sentiments.get(sentiment["polarity"])!=None:
		x = sentiments.get(sentiment["polarity"])
		sentiments[sentiment["polarity"]] = x + 1
	else:
                sentiments[sentiment["polarity"]] = 1

