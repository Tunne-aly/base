import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Doc2Vec

input_size, hidden_size, output_size = 10, 10,2

n_n = torch.nn.Sequential(
          # Linear applies linear transformation y=Ax+b
          torch.nn.Linear(input_size, hidden_size),
          # ReLU is one of the non-linear activations
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_size, output_size),
          # Use softmax last to achieve probabilities
          torch.nn.Softmax(),
        )

# mean(xn-yn)^2
loss_function = torch.nn.MSELoss()

# the label has 2 dimensions: negative and positive
data = [
  dict([('value', torch.Tensor(input_size)),
    ('label', torch.Tensor([0,1]))]),
  dict([('value', torch.Tensor(input_size)),
    ('label', torch.Tensor([1,0]))])
  ]
def train():
# let's train 100 times because it's a pretty number
  for i in range(1000):
    for d in data:
      # pycharm mostly understands Variables
      pred = n_n(Variable(d['value']))
      loss = loss_function(pred, Variable(d['label']))
      # the gradients have to be reset because they're cumulative
      n_n.zero_grad()
      # backpropagate
      loss.backward()
      # in gradient descent the size of the "step" taken
      # to the direction on of the opposite of the gradient for a parameter
      step_size = 0.0001

      for param in n_n.parameters():
        # update parameters
        param.data -= step_size * param.grad.data

# def prepare_data(data):
#   # use Doc2Vec to represent sentences as vectors
#   for d in data:
#     Doc2Vec([], size=input_size, window=8, min_count=3)
