import re, sys
from os.path import isfile
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd
import nn
from functools import reduce

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

def plot_sentences(data_path):
    data = []
    labels = []
    _, train_data_loader = nn.get_data_as_sentence_tensors(data_path)
    print('plotting training data')
    for datum in train_data_loader:
        sentence_token = datum[1].view(-1, reduce(lambda z, y: z * y, datum[1].size()[1:], 1))
        token = sentence_token[0].numpy()
        labels.append(datum[0])
        data.append(token)

    # TSNE
    tsne_model = TSNE(n_components=2, init='pca')
    new_values = tsne_model.fit_transform(data)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    label_colors = ['black', 'red', 'blue', 'green', 'lightgreen']

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i], c=label_colors[labels[i]])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points')
    plt.show()

removable_words = ['ja', 'mutta', 'on', 'ole', 'oli', 'kun', 'kuin', 'myös']

def plot_words(data_path, amount_of_words, perplexity):
    print('Reading sentences from file...')
    sentences = nn.get_sentences(sys.argv[1])
    print('Read {} sentences'.format(len(sentences)))
    data = []
    labels = []
    print('Calculating new embeddings...')
    cleaned_sentences = [list(map(lambda x: re.sub("[^a-öA-Ö0-9 ']", '', x.strip().lower()), s[1])) for s in sentences]
    print('filtering out {}'.format(removable_words))
    model = Word2Vec([list(filter(lambda x: x != '' and x not in removable_words, sentence)) for sentence in cleaned_sentences], size=300, window=10, workers=2, sorted_vocab=True)
    print('Done')

    print('Plotting:')
    for i in range(amount_of_words):
        word = model.wv.index2word[i]
        print(word)
        data.append(model[word])
        labels.append(word)

    print('plotting {} most common words'.format(len(data)))

    print('similarity kiva hyvä {}'.format(model.wv.similarity('kiva', 'hyvä')))
    print('similarity suosittelen hyvä {}'.format(model.wv.similarity('suosittelen', 'hyvä')))
    print('similarity huono hyvä {}'.format(model.wv.similarity('huono', 'hyvä')))
    print('similarity samsung hyvä {}'.format(model.wv.similarity('samsung', 'hyvä')))
    print('similarity ei kyllä {}'.format(model.wv.similarity('ei', 'kyllä')))
    print('similarity erinomainen pettymys {}'.format(model.wv.similarity('erinomainen', 'pettymys')))
    print('similarity plussaa suositella {}'.format(model.wv.similarity('plussaa', 'suositella')))

    del model
    tsne_model = TSNE(perplexity=perplexity,  n_components=2, init='pca', n_iter=4000)
    new_values = tsne_model.fit_transform(data)

    print('tsne model done')

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(100, 100))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(4, 1),
                     textcoords='offset points')
    plt.show()


if len(sys.argv) < 2:
    print('Please specify data path')
    exit()
if len(sys.argv) == 3:
    print('Please specify perplexity with the amount of plottable words')
    exit()
amount_of_words = sys.argv[2] if len(sys.argv) > 2 else 150
perplexity = sys.argv[3] if len(sys.argv) > 3 else 50
plot_words(sys.argv[1], int(amount_of_words), int(perplexity))
