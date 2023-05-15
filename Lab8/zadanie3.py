# pip3 install --upgrade git+https://github.com/JustAnotherArchivist/snscrape.git

# import os

# os.system("snscrape --jsonl --max-results 100 twitter-search '#ukraine' > ukraine_tweets.json")
# os.system("snscrape --jsonl --max-results 50 twitter-search '#Gdańsk since:2020-06-01 until:2020-07-31' > gdansk_tweets.json")

import pandas as pd

ukraine_tweets_df = pd.read_json('ukraine_tweets.json', lines=True, encoding='utf-16')
gdansk_tweets_df = pd.read_json('gdansk_tweets.json', lines=True, encoding='utf-16')

print(ukraine_tweets_df.rawContent)
print(gdansk_tweets_df.rawContent)
