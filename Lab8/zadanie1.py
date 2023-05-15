import string

import nltk
import matplotlib.pyplot as plt

nltk.download([
    'punkt',
    'stopwords',
    'wordnet'
])

stopwords = nltk.corpus.stopwords.words("english")
stopwords.append("'s")
stopwords.append("n't")
stopwords.append("â€")
stopwords.append('“')

with open('bbc_article.txt') as file:
    text = file.read()

print(len(nltk.word_tokenize(text)))
words = nltk.word_tokenize(text)

words = [w for w in words if w.lower() not in stopwords]
words = [w for w in words if w.lower() not in string.punctuation]

print(len(words))

lem_tokens = set()
wnl = nltk.WordNetLemmatizer()

for word in words:
    lem_tokens.add(wnl.lemmatize(word))

print(len(lem_tokens))

counter = nltk.Counter(words)
top_N = counter.most_common(10)
for token, count in top_N:
    print(token, ":", count)
labels, values = zip(*top_N)
plt.plot(labels, values)
plt.xticks(rotation=45)
plt.savefig("tokens.png")
plt.show()

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=200, stopwords=stopwords).generate(text)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud.png')
plt.show()
