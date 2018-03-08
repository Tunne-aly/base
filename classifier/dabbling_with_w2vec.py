import re, sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import pandas as pd
import sa
from functools import reduce

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

def plot_tsne(data, labels):
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


if len(sys.argv[1]) < 2:
    print('Please specify embeddings path')
    exit()
train_data_loader, test_data_loader = sa.get_data_as_sentence_tensors(sys.argv[1])
data = []
labels = []
for datum in train_data_loader:
    sentence_token = datum[1].view(-1, reduce(lambda z, y: z * y, datum[1].size()[1:], 1))
    token = sentence_token[0].numpy()
    labels.append(datum[0])
    print(token)
    data.append(token)
plot_tsne(data, labels)
