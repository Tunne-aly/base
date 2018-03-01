# reviews from http://jmcauley.ucsd.edu/data/amazon/

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

input_size, hidden_size, output_size, batch_size, epoch_amount, train_set_size = 100, 200, 5, 200, 10, 0.8

def label_as_one_hot(grade):
  vector = torch.Tensor(output_size).zero_()
  vector[grade - 1] = 1
  return vector

def label_as_binary_one_hot(grade):
  vector = torch.Tensor(output_size).zero_()
  if grade <= 1:
    vector[0] = 1
  else:
    vector[1] = 1
  return vector

def get_prediction(output):
  max_index = 0
  for i in range(output_size):
    if output[i].data[0] > output[max_index].data[0]:
      max_index = i
  return max_index + 1

def train(n_n, vectors, labels):
  loss_function = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(n_n.parameters(), lr=0.001)
  for j in range(epoch_amount):
    i = 0
    while i + batch_size < len(vectors):
      random_indexes = torch.randperm(batch_size) + i
      batch = vectors[random_indexes].float()
      batch_labels = labels[random_indexes].float()
      preds = n_n(Variable(batch))
      print('TRAINING preds {}'.format(preds))
      print('TRAINING labels {}'.format(batch_labels))
      loss = loss_function(preds, Variable(batch_labels))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      i += batch_size

def test(n_n, vectors, labels):
  success = 0
  i = 0
  while i + batch_size < len(vectors):
    indexes = torch.arange(batch_size).long() + i
    batch = vectors[indexes].float()
    batch_labels = labels[indexes].float()
    preds = n_n(Variable(batch))
    print('TESTING labels: {}'.format(batch_labels))
    print('TESTING prediction: {}'.format(preds))
    grades = torch.stack(list(map(lambda x: label_as_one_hot(get_prediction(x)), preds)), 0)
    print('TESTING predictions as grades : {}'.format(grades))
    success = grades.eq(batch_labels).sum()
    i += batch_size
  print('failures ', str(len(vectors) - success))
  return success/len(vectors)

def prepare_data_doc2vec(input_size):
  i = 0
  texts =  list(utils.get_review_text_from_zip(sys.argv[1]))
  random.shuffle(texts)
  data_set_size = 7000*5
  vectors = np.zeros([data_set_size, input_size], dtype=float)
  labels = np.zeros([data_set_size, output_size], dtype=float)
  set_divider = int(train_set_size*data_set_size)
  data_freqs = dict([(1, 0), (2,0), (3,0), (4,0), (5,0)])
  for t in texts:
    grade = int(t[1])
    review = t[0]
    words = utils.normalize_words(review)
    tag = str(grade)
    if len(words) == 0:
      continue
    if data_freqs[grade] == 7000:
      continue
    doc = Doc2Vec([doc2vec.TaggedDocument(words, [tag])], size=input_size, window=5, min_count=1)
    vector = torch.from_numpy(doc.docvecs[tag])
    vectors[i] = vector.float()
    labels[i] = label_as_one_hot(grade)
    data_freqs[grade] += 1
    i += 1
  print('data frequencies {}'.format(data_freqs))
  return (torch.from_numpy(vectors[:set_divider]), torch.from_numpy(labels[:set_divider]),
          torch.from_numpy(vectors[set_divider:]), torch.from_numpy(labels[set_divider:]))

if len(sys.argv) < 2:
  print('Please give path to data file as an argument')
  exit()

whole_start_time = time.time()
print('preparing data')
start_time = time.time()
(train_vectors, train_labels, test_vectors, test_labels) = prepare_data_doc2vec(input_size)
print('preparing data took ' + str(time.time() - start_time) + ' seconds')
network = torch.nn.Sequential(
          torch.nn.Linear(input_size, hidden_size),
          torch.nn.Sigmoid(),
          torch.nn.Linear(hidden_size, hidden_size),
          torch.nn.Sigmoid(),
          torch.nn.Linear(hidden_size, int(output_size)),
          torch.nn.Softmax(dim=1),
      )
print('starting training with a training set of size {}'.format(len(train_vectors)))
start_time = time.time()
train(network, train_vectors, train_labels)
print('training data took ' + str(time.time() - start_time) + ' seconds')
print('starting testing with a testing set of size {}'.format(len(test_vectors)))
success_ratio = test(network, test_vectors, test_labels)
print('whole procedure took ', str(time.time() - whole_start_time))
print('success ratio ', str(success_ratio))
