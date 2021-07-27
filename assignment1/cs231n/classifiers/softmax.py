from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        # 수가 너무 커지는 것을 방지
        scores -= np.max(scores)

        # softmax 함수 정의
        softmax = lambda scores, index: np.exp(scores[index]) / np.sum(np.exp(scores))

        # loss에 정답 클래스의 softmax 값을 -log취한 것을 더함
        loss += -np.log(softmax(scores, y[i]))

        # gradient 부분
        for j in range(num_classes):
            dW[:,j] += softmax(scores, j) * X[i]
        dW[:,y[i]] -= X[i]

    loss /= num_train
    dW /= num_train

    # regularization 부분
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1).reshape(num_train, 1)
    
    # loss 부분
    total_scores_exp = np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
    softmax = np.exp(scores) / total_scores_exp
    loss = np.sum(-np.log(softmax[np.arange(num_train), y]))

    # gradient 부분
    # 정답 클래스에서의 y_i는 1이기 때문에 이를 다 빼준 후에 X를 곱한다.
    softmax[np.arange(num_train), y] -= 1
    dW = X.T.dot(softmax)

    loss /= num_train
    dW /= num_train

    # regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
