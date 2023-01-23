#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os
import math
import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):

    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q3.1a
        y_hat = np.argmax(self.W.dot(x_i))
        if y_hat != y_i:
            # Perceptron update.
            self.W[y_i, :] += x_i # Right Class
            self.W[y_hat, :] -= x_i # Wrong Class
        return
        raise NotImplementedError


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q3.1b
        # Multi-class Logistic Regression        
        label_scores = self.W.dot(x_i)[:, None] # Label scores according to the model (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1)) # One-hot vector with the true label (num_labels x 1).
        y_one_hot[y_i] = 1
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))# Softmax function.
        self.W += learning_rate * (y_one_hot - label_probabilities) * x_i[None, :] # SGD update
        return
        raise NotImplementedError


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.w1 = np.random.normal(0.1, 0.1, size=(hidden_size,n_features))
        self.w2 = np.random.normal(0.1, 0.1, size=(n_classes,hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(n_classes)
        return
        raise NotImplementedError

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        z1 = X @ self.w1.T + self.b1
        X_2 = X_2 = np.maximum(z1, 0) # ReLU activation function
        out = X_2 @ self.w2.T + self.b2
        y_hat = out.argmax(axis=1)

        return y_hat 
        raise NotImplementedError

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]

        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001): # Update Weights

        for x, y in zip(X, y):

            # One Hot encoding of Y
            y_oh = np.zeros(self.w2.shape[0])
            y_oh[y] = 1

            ## Forward Pass ##
            z1 = self.w1.dot(x) + self.b1
            X_2 = X_2 = np.maximum(z1, 0) # ReLU activation function
            out = self.w2.dot(X_2) + self.b2 # the score of each class
        
            ## Softmax ##
            out = out - out.max() # Normalization to prevent overflow (Question @50, INSTRUCTOR IN PIAZZA)
            probs = np.exp(out) / np.sum(np.exp(out)) #softmax
            
            ## Backwards pass ##
            grad_z2 = probs - y_oh # Cross-Entropy 
            grad_w2 = grad_z2[:, None].dot(X_2[:, None].T) # Chain Rule wrt W
            grad_b2 = grad_z2 # Chain Rule wrt b

            grad_h1 = self.w2.T.dot(grad_z2)
            grad_z1 = np.where(X_2>0, grad_h1, 0)
            grad_w1 = grad_h1[:, None].dot(x[:, None].T) # Chain Rule wrt W
            grad_b1 = grad_z1 # Chain Rule wrt b

            ## Gradient updates ##
            eta = learning_rate
            self.w1 -= eta*grad_w1
            self.b1 -= eta*grad_b1
            self.w2 -= eta*grad_w2
            self.b2 -= eta*grad_b2

        return
        raise NotImplementedError


def plot(epochs,train_accs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
   #plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"

    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    train_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(train_X,train_y,learning_rate=opt.learning_rate)
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, train_accs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
