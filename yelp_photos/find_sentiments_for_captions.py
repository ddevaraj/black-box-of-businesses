import json
from aylienapiclient import textapi
  
app_id  = "40766b3c"
app_key = "23a9622cb9c7bf182bdf2c3fe58d0dff"

client = textapi.Client(app_id, app_key)

data = []

with open('../yelp_academic_dataset_photo.json') as f:
        for line in f:
                data.append(json.loads(line))

sentiments = {}

for i in data:
	caption_text = i["caption"]
	sentiment = client.Sentiment({'text': caption_text})
        if sentiment["polarity"]=="Positive" or sentiment["polarity"]=="Negative":
                print("Caption_text is : "+caption_text+" polarity : "+sentiment["polarity"])
        if sentiments.get(sentiment["polarity"])!=None:
		x = sentiments.get(sentiment["polarity"])
		sentiments[sentiment["polarity"]] = x + 1
	else:
                sentiments[sentiment["polarity"]] = 1

