import re
import os
import json
import json_lines
import numpy as np
import pandas as pd
from transformers import pipeline

from datetime import datetime
from pysentimiento.preprocessing import preprocess_tweet

format = '%a %b %d %X +0000 %Y'

model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

translate = {
    "Positive": "POS",
    "Neutral": "NEU",
    "Negative": "NEG"
}

#def clean_string(tweet):
#    return re.sub(r'http\S+', '', re.sub(r'@\S+', '', tweet.replace('\n', '')).replace('#', ''))

#def predict_sentiment(clean_tweet):
#    sentiment = analyzer.predict(clean_tweet)
#    return max(sentiment.probas.items(), key=lambda x: x[1])

def predict_sentiment(clean_tweet):
    return sentiment_task(clean_tweet)[0]

#def predict_emotion(clean_tweet):
#    emotion = emotion_analyzer.predict(clean_tweet)
#    return max(emotion.probas.items(), key=lambda x: x[1])

def run_conversion(current_tweet_id):

    current_tweet = {}
    current_tweet_dataset = dataset[dataset['id'] == current_tweet_id]
    current_tweet["created_at"] = "Fri Feb 22 12:01:52 +0000 2019"#current_tweet_dataset["created_at"].values[0]
    current_tweet["id"] = current_tweet_id
    current_tweet["id_str"] = str(current_tweet_id)
    current_tweet["full_text"] = current_tweet_dataset["tweet"].values[0]
    current_tweet["user"] = {}
    current_tweet["user"]["id"] = current_tweet_dataset["user_id"].values[0]
    current_tweet["user"]["id_str"] = str(current_tweet_dataset["user_id"].values[0])
    current_tweet["user"]["name"] = current_tweet_dataset["name"].values[0]
    current_tweet["user"]["screen_name"] = current_tweet_dataset["username"].values[0]
    sentiment = predict_sentiment(preprocess_tweet(current_tweet_dataset["tweet"].values[0]))
    current_tweet["sentiment"] = {}
    current_tweet["sentiment"]["sentiment"] = translate[sentiment["label"]]
    current_tweet["sentiment"]["probability"] = sentiment["score"]
#    emotion = predict_emotion(preprocess_tweet(current_tweet_dataset["tweet"].values[0]))
#    current_tweet["emotion"] = {}
#    current_tweet["emotion"]["emotion"] = emotion[0]
#    current_tweet["emotion"]["probability"] = emotion[1]
    if current_tweet_dataset["user_followers"].values[0] is np.NaN:
        current_tweet["user"]["followers_count"] = current_tweet_dataset["user_followers"].values[0]
        current_tweet["user"]["friends_count"] = current_tweet_dataset["user_following"].values[0]
        current_tweet["user"]["verified"] = current_tweet_dataset["user_verified"].values[0]
    else:
        current_tweet["user"]["followers_count"] = 0
        current_tweet["user"]["friends_count"] = 0
        current_tweet["user"]["verified"] = False
    with open('./data/pt/'+ str(current_tweet_id) + '.jsonl', 'r') as f:   
        for twarc in (json_lines.reader(f)):
            #twarc = json.load(json_file)
            for response in twarc['data']:
                if response["id"] == current_tweet_id:
                    if user["public_metrics"]:
                        current_tweet["user"]["followers_count"] = user["public_metrics"]["followers_count"]
                        current_tweet["user"]["friends_count"] = user["public_metrics"]["following_count"]
                        current_tweet["user"]["verified"] = user["verified"]
                if response["id"] != current_tweet_id:
                    reply = {}
                    reply["created_at"] = datetime.strptime(response["created_at"], '%Y-%m-%dT%H:%M:%S.000Z').strftime(format)
                    reply["id"] = int(response["id"])
                    reply["id_str"] = response["id"]
                    reply["full_text"] = response["text"]
                    reply_sentiment = predict_sentiment(preprocess_tweet(response["text"]))
                    reply["sentiment"] = {}
                    reply["sentiment"]["sentiment"] = translate[reply_sentiment["label"]]
                    reply["sentiment"]["probability"] = reply_sentiment["score"]
#                    reply_emotion = predict_emotion(preprocess_tweet(response["text"]))
#                    reply["emotion"] = {}
#                    reply["emotion"]["emotion"] = reply_emotion[0]
#                    reply["emotion"]["probability"] = reply_emotion[1]
                    author = response["author_id"]
                    reply["user"] = {}
                    for user in twarc["includes"]["users"]:
                        if user["id"] == author:
                            reply["user"]["id"] = int(user["id"])
                            reply["user"]["id_str"] = user["id"]
                            reply["user"]["name"] = user["name"]
                            reply["user"]["screen_name"] = user["username"]
                            if user["public_metrics"]:
                                reply["user"]["followers_count"] = user["public_metrics"]["followers_count"]
                                reply["user"]["friends_count"] = user["public_metrics"]["following_count"]
                                reply["user"]["verified"] = user["verified"]
                            else:
                                reply["user"]["followers_count"] = 0
                                reply["user"]["friends_count"] = 0
                                reply["user"]["verified"] = False
                    placeholder = current_tweet.copy();
                    placeholder["retweeted_status"] = reply
                    
                    savefile = open("./data/pt-2021_4.jsonl", "a")
                    savefile.write(json.dumps(placeholder, separators=(',', ':'), cls=NpEncoder))
                    savefile.write("\n")
                    savefile.close()

dataset = pd.read_csv("./data/por_2021.csv", low_memory=False)
#dataset['created_at'] = pd.to_datetime(dataset['created_at'], format='%Y-%m-%d T%H:%M:%S.000Z')
directory = './data/pt'
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    tweet_id = int(os.path.splitext(filename)[0])
    # checking if it is a file
    if len(dataset[dataset['id'] == tweet_id]) > 0:
        if os.path.isfile(f):
            run_conversion(tweet_id)

