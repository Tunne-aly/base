# https://jlorince.github.io/viz-tutorial/

import re, sys
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Doc2Vec, doc2vec
import pandas as pd
import numpy as np
import math, sys, random, time
import traceback
import torch
import utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Doc2Vec, doc2vec
import pdb

input_size, hidden_size, output_size, batch_size, epoch_amount, train_set_size = 100, 200, 6, 200, 10, 0.8

def prepare_data_doc2vec(input_size):
  i = 0
  texts =  list(utils.get_review_text_from_zip(sys.argv[1]))
  random.shuffle(texts)
  data_set_size = len(texts)
  vectors = np.zeros([data_set_size, input_size], dtype=float)
  labels = np.empty(data_set_size)
  set_divider = int(train_set_size*data_set_size)
  for t in texts:
    grade = int(t[1])
    review = t[0]
    words = utils.normalize_words(review)
    tag = str(grade)
    if len(words) == 0:
      continue
    doc = Doc2Vec([doc2vec.TaggedDocument(words, [tag])], size=input_size, window=5, min_count=1)
    vector = torch.from_numpy(doc.docvecs[tag])
    vectors[i] = vector.float()
    labels[i] = grade
    i += 1
  return (torch.from_numpy(vectors[:set_divider]), labels[:set_divider],
          torch.from_numpy(vectors[set_divider:]), labels[set_divider:])

(train_vectors, train_labels, test_vectors, test_labels) = prepare_data_doc2vec(100)

pca = PCA(n_components=2)
transformed = pd.DataFrame(pca.fit_transform(test_vectors, test_labels))
plt.scatter(transformed[test_labels==1][0], transformed[test_labels==1][1], label='Grade 1', c='red')
plt.scatter(transformed[test_labels==2][0], transformed[test_labels==2][1], label='Grade 2', c='blue')
plt.scatter(transformed[test_labels==3][0], transformed[test_labels==3][1], label='Grade 3', c='red')
plt.scatter(transformed[test_labels==4][0], transformed[test_labels==4][1], label='Grade 4', c='green')
plt.scatter(transformed[test_labels==5][0], transformed[test_labels==5][1], label='Grade 5', c='lightgreen')
plt.legend()
plt.show()
