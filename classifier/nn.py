# reviews from http://jmcauley.ucsd.edu/data/amazon/

import numpy as np
import math, sys, random
import traceback
import torch
import utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Doc2Vec, doc2vec
import pdb

input_size, hidden_size, output_size = 100, 200, 6

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
  loss = Variable(torch.Tensor([1]))
  while loss.data[0] > 0.001:
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
          print('TRAINING pred vector for ' + str(label)+ ' in testing ' + str(pred))
          print('')
          loss = loss_function(pred, Variable(label))
          # the gradients have to be reset because they're cumulative
          n_n.zero_grad()
          # backpropagate
          loss.backward()
          # in gradient descent the size of the "step" taken
          # to the opposite direction of the gradient for a parameter
          step_size = 0.00001
          for param in n_n.parameters():
            # update parameters i.e. weights & biases
            param.data -= step_size * param.grad.data
        except KeyError:
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
        print('TESTING pred vector for ' + str(pred)+ ' in testing ' + str(pred_vector))
        print('')
        if pred == grade:
          success += 1
        # else:
          # print('pred vector for ' + str(label)+ ' in testing ' + str(pred_vector))
        total += 1
      except KeyError:
        # if tag was not found in dictionary
        # traceback.print_exc()
        continue
  print('failures ' + str(total - success))
  if total == 0:
    return 0
  return success/total

def prepare_data(input_size):
  train_document = []
  test_document = []
  # let's assume that the scale goes from 1 to 5
  train_grade_freq = dict([(0,0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])
  test_grade_freq = dict([(0,0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])
  review_lines = list(utils.get_review_text_from_zip(sys.argv[1]))
  random.shuffle(review_lines)
  for l in review_lines:
    # l[1] contains grade, l[0] the review
    grade = l[1]
    review = l[0]
    words = utils.normalize_words(review)
    tag = ''
    # we'll tag the review so we'll know its label in training
    # try to divide the emotions equally to training and testing
    if grade == 4 or grade == 3 or grade == 2:
      continue
    if grade == 5 and random.randint(0, 9) > 0:
      continue
    if (train_grade_freq[grade] > test_grade_freq[grade]):
      tag = str(int(grade))+ '-' + str(test_grade_freq[grade])
      test_grade_freq[grade] += 1
      test_document.append(doc2vec.TaggedDocument(words, [tag]))
    else:
      tag = str(int(grade))+ '-' + str(train_grade_freq[grade])
      train_grade_freq[grade] += 1
      train_document.append(doc2vec.TaggedDocument(words, [tag]))
    # print(tag)
  # use Doc2Vec to represent reviews as vectors
  return (Doc2Vec(train_document, size=input_size, window=8, min_count=3),
          Doc2Vec(test_document, size=input_size, window=8, min_count=3),
          train_grade_freq, test_grade_freq)

if len(sys.argv) < 2:
  print('Please give path to data file as an argument')
  exit()

print('preparing data')
(train_data, test_data, train_grade_freq, test_grade_freq) = prepare_data(input_size)
network = torch.nn.Sequential(
          # Linear applies linear transformation y=Ax+b
          torch.nn.Linear(input_size, hidden_size),
          # use a non-linear activations
          torch.nn.Sigmoid(),
          torch.nn.Sigmoid(),
          torch.nn.Linear(hidden_size, int(output_size)),
          # Use softmax last to achieve probabilities
          torch.nn.Softmax(dim=0),
      )
print('starting training')
train(network, train_data, train_grade_freq)
print('starting testing')
success_ratio = test(network, test_data, test_grade_freq)
print('success ratio ', success_ratio)
print('test grade frequencies: ', test_grade_freq)
print('train grade frequencies: ', train_grade_freq)
