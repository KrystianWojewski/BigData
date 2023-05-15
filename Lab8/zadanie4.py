# snscrape --jsonl --max-results 250 twitter-search '#HIMYM unpopular opinion lang:en' > HIMYM_tweets.json
# snscrape --jsonl --max-results 250 twitter-search '#HIMYM popular opinion lang:en' >> HIMYM_tweets.json

import string

import nltk
import matplotlib.pyplot as plt

import pandas as pd

nltk.download([
    'punkt',
    'stopwords',
    'wordnet'
])

stopwords = nltk.corpus.stopwords.words("english")

HIMYM_tweets_df = pd.read_json('HIMYM_tweets.json', lines=True, encoding='utf-16')

# print(HIMYM_tweets_df.rawContent)

tokens = []
text = ""

for tweet in HIMYM_tweets_df.rawContent:
    text += str(tweet).lower()
    text += " "
    words = nltk.word_tokenize(tweet)
    for w in words:
        tokens.append(w.lower())

print(len(tokens))

tokens = [t for t in tokens if t not in stopwords]
tokens = [t for t in tokens if t not in string.punctuation]

print(len(tokens))

lem_tokens = set()
wnl = nltk.WordNetLemmatizer()

for token in tokens:
    lem_tokens.add(wnl.lemmatize(token))

print(len(lem_tokens))

counter = nltk.Counter(tokens)
top_N = counter.most_common(10)
for token, count in top_N:
    print(token, ":", count)
labels, values = zip(*top_N)
plt.plot(labels, values)
plt.xticks(rotation=45)
plt.savefig("tokens4.png")
plt.show()

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=200, stopwords=stopwords).generate(text)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud4.png')
plt.show()

import text2emotion as te

tweets_emotions_list = []

for tweet in HIMYM_tweets_df.rawContent:
    tweets_emotions_list.append(te.get_emotion(tweet))

tweets_emotions_dict = {'Happy': 0.0, 'Angry': 0.0, 'Surprise': 0.0, 'Sad': 0.0, 'Fear': 0.0}

for tweet in tweets_emotions_list:
    tweets_emotions_dict["Happy"] += tweet["Happy"]
    tweets_emotions_dict["Angry"] += tweet["Angry"]
    tweets_emotions_dict["Surprise"] += tweet["Surprise"]
    tweets_emotions_dict["Sad"] += tweet["Sad"]
    tweets_emotions_dict["Fear"] += tweet["Fear"]

tweets_emotions_dict["Happy"] = tweets_emotions_dict["Happy"]/len(tweets_emotions_list)
tweets_emotions_dict["Angry"] = tweets_emotions_dict["Angry"]/len(tweets_emotions_list)
tweets_emotions_dict["Surprise"] = tweets_emotions_dict["Surprise"]/len(tweets_emotions_list)
tweets_emotions_dict["Sad"] = tweets_emotions_dict["Sad"]/len(tweets_emotions_list)
tweets_emotions_dict["Fear"] = tweets_emotions_dict["Fear"]/len(tweets_emotions_list)

print(tweets_emotions_dict)

wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(tweets_emotions_dict)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('emotions_wordcloud4.png')
plt.show()
