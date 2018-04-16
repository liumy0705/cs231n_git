import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in xrange(num_train):
        s = X[i].dot(W)
        scores = s - max(s)    # instead: first shift the values of scores so that the highest number is 0
        scores_E = np.exp(scores)
        Z = np.sum(scores_E)
        scores_target = scores_E[y[i]]
        loss += -np.log(scores_target / Z)
        for j in xrange(num_class):
            if j == y[i]:
                dW[:, j] += -(1 - scores_E[j] / Z) * X[i]
            else:
                dW[:, j] += X[i] * scores_E[j] / Z
  loss = loss / num_train + reg * np.sum(W * W)
  dW = dW / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_train = X.shape[0]
  s = np.dot(X, W)    # N*C
  scores = s - np.max(s, axis = 1, keepdims = True)    # N*C
  scores_E = np.exp(scores)    # N*C
  Z = np.sum(scores_E, axis = 1, keepdims = True)    # N*C
  prob = scores_E / Z    # N*C
  y_trueClass = np.zeros_like(prob)
  y_trueClass[range(num_train), y] = 1.0 #Set the actual class to 1 and the rest continue to be 0
  loss += -np.sum(y_trueClass * np.log(prob)) / num_train + reg * np.sum(W * W)
  dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

