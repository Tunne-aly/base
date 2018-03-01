# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

f = open('lolcat_wiki.txt', 'r')
sentences = []
for line in f:
    words = list(map(lambda x: re.sub("[^a-zA-Z0-9 ']", '', x.strip().lower()), line.split(' ')))
    sentences.append(words)
model = Word2Vec(sentences, min_count=2)

# summarizes the model
print(model)
print('size = the amount of nn layers used')

# print the vocabulary
print(list(model.wv.vocab))

print('\n .........the interesting stuff........\n')

print('lol: \n', model['lol'])
print('cat: \n', model['cat'])
print('meme: \n', model['meme'])
# throws an error if a model for a word not in the vocab is searched
# print(model['soviet'])

print('\n')

print("print(model.wv.similarity('lolcat', 'cat')): ")
print(model.wv.similarity('lolcat', 'cat'))

print("print(model.wv.similarity('cat', 'cat')): ")
print(model.wv.similarity('cat', 'cat'))

print("print(model.wv.similarity('meme', 'image')): ")
print(model.wv.similarity('meme', 'image'))

print("print(model.wv.most_similar(positive=['meme'], negative=['cat'])):")
print(model.wv.most_similar(positive=['meme'], negative=['cat']))

print('\n')

x = model[model.wv.vocab]
# Principal Component Analysis
pca = PCA(n_components=2)
result = pca.fit_transform(x)
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
