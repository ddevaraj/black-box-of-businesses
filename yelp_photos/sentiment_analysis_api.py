from aylienapiclient import textapi

app_id  = "40766b3c"
app_key = "23a9622cb9c7bf182bdf2c3fe58d0dff"

client = textapi.Client(app_id, app_key)

sentiment = client.Sentiment({'text': 'John is a very good football player!'})

print(sentiment)
