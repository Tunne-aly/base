# reviews from http://jmcauley.ucsd.edu/data/amazon/

import numpy as np
import math
import traceback
import torch
import utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Doc2Vec, doc2vec
import pdb

input_size, hidden_size, training_data_size, output_size = 100, 100, 0.8, 6

def label_as_one_hot(grade):
  vector = torch.Tensor(output_size).zero_()
  vector[grade] = 1
  return vector

def get_prediction(output):
  max_index = 0
  for i in range(output_size):
    # they're Variables
    if output[i].data[0] > output[max_index].data[0]:
      max_index = i
  return max_index

def train(n_n, data, grade_freq):
  # mean(x_n-y_n)^2 as the loss function
  loss_function = torch.nn.MSELoss()
  # let's train 100 times because it's a pretty number
  for t in range(100):
    # range(6) gives 0...5
    for grade in range(output_size):
      # frequency gives how many reviews with this grade there are
      frequency = grade_freq[grade]
      label = label_as_one_hot(grade)
      for i in range(frequency):
        tag = str(grade) + '-' + str(i)
        try:
          vector = data.docvecs[tag]
          pred = n_n(Variable(torch.from_numpy(vector)))
          loss = loss_function(pred, Variable(label))
          # the gradients have to be reset because they're cumulative
          n_n.zero_grad()
          # backpropagate
          loss.backward()
          # in gradient descent the size of the "step" taken
          # to the opposite direction of the gradient for a parameter
          step_size = 0.0001
          for param in n_n.parameters():
            # update parameters
            param.data -= step_size * param.grad.data
        except:
          # if tag was not found in dictionary
          # traceback.print_exc()
          continue

def test(n_n, data, grade_freq):
  success = 0
  total = 0
  for grade in range(output_size):
    # frequency gives how many reviews with this grade there are
    frequency = grade_freq[grade]
    label = label_as_one_hot(grade)
    for i in range(frequency):
      tag = str(grade) + '-' + str(i)
      try:
        vector = data.docvecs[tag]
        pred_vector = n_n(Variable(torch.from_numpy(vector)))
        pred = get_prediction(pred_vector)
        print(pred)
        if pred == grade:
          success += 1
        total += 1
      except:
        # if tag was not found in dictionary
        # traceback.print_exc()
        continue
  return success/total

def prepare_data(training_data_size, input_size):
  document = []
  # let's assume that the scale goes from 1 to 5
  grade_freq = dict([(0,0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])
  for l in utils.get_review_text_from_zip('reviews_Amazon_Instant_Video_5.json.gz'):
    # l[1] contains grade, l[0] the review
    grade = l[1]
    review = l[0]
    words = utils.normalize_words(review)
    # we'll tag the review so we'll know its label in training
    tag = str(int(grade))+ '-' + str(grade_freq[grade])
    # print(tag)
    grade_freq[grade] += 1
    document.append(doc2vec.TaggedDocument(words, [tag]))
  # use Doc2Vec to represent reviews as vectors
  training_index = math.floor(len(document)*training_data_size)
  return (Doc2Vec(document[:training_index], size=input_size, window=8, min_count=3),
          Doc2Vec(document[training_index+1:], size=input_size, window=8, min_count=3),
          grade_freq)

(train_data, test_data, grade_freq) = prepare_data(training_data_size, input_size)
network = torch.nn.Sequential(
          # Linear applies linear transformation y=Ax+b
          torch.nn.Linear(input_size, hidden_size),
          # ReLU is one of the non-linear activations
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_size, int(output_size)),
          # Use softmax last to achieve probabilities
          torch.nn.Softmax(),
      )
print('starting training')
train(network, train_data, grade_freq)
print('starting testing')
success_ratio = test(network, test_data, grade_freq)
print('success ratio ' + success_ratio)
