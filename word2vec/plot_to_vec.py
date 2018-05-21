import re, sys
from os.path import isfile
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd
import nn
import csv
from functools import reduce
from random import shuffle
import torch
from os import listdir, sep

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

EMBEDDINGS_PATH = "embeddins.bin"
BLOCK_LENGTH = 20
TEST_DATA_SIZE = 300

def get_sentence_tensor(sentence, embeddings, block_length):
    return torch.stack([torch.stack([get_embedding(i, sentence, embeddings) for i in range(block_length)])])


def get_embedding(idx, sentence, embeddings):
    return (torch.from_numpy(embeddings[sentence[idx]])
            if len(sentence) > idx and sentence[idx] in embeddings
            else torch.zeros(300))


def get_sentences(data_path):
    ls = []
    for p in listdir(data_path):
        if p.endswith(".csv"):
            for sent in read_sentences("{}{}{}".format(data_path, sep, p)):
                ls.append(sent)
    return ls

def read_sentences(file_path):
    with open(file_path) as infile:
        reader = csv.reader(infile)
        for line in reader:
          yield line[1], [re.sub("[^a-öA-Ö ']", '', w).strip().lower() for w in line]

def get_data_as_sentence_tensors(data_path=None, sentences=None, relearn=False, embeddings_path=EMBEDDINGS_PATH, block_length=BLOCK_LENGTH, test_data_size=TEST_DATA_SIZE, normalize=False):
    if not sentences:
        print("Reading sentences from file...")
        sentences = get_sentences(data_path)
        print("Read {} sentences".format(len(sentences)))
    if isfile(embeddings_path) and not relearn:
        print("Retreiving embeddings from {}".format(embeddings_path))
        embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
    else:
        print("Calculating new embeddings...")
        model = Word2Vec([t[1] for t in sentences], size=300, window=10, workers=2)
        print("Done")
        embeddings = model.wv
        del model
        embeddings.save_word2vec_format(embeddings_path, binary=True)
        print("Embeddings stored to {}".format(embeddings_path))
    print("Building sentece tensors...")
    data_freqs = dict([(1,0), (2,0), (3,0), (4,0), (5,0)])
    cleaned_sentences = []
    for sentence in sentences:
        grade = int(sentence[0])
        cleaned_sentences.append((sentence[0], sentence[1]))
        data_freqs[grade] += 1
    print(data_freqs)
    shuffle(cleaned_sentences)
    test_data = [(int(s[0]) - 1, get_sentence_tensor(s[1], embeddings, block_length)) for s in cleaned_sentences[:test_data_size]]
    train_data = [(int(s[0]) - 1, get_sentence_tensor(s[1], embeddings, block_length)) for s in cleaned_sentences[test_data_size:]]
    del cleaned_sentences, sentences
    return test_data, train_data

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
