#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains two classifiers: Naive Bayes and Logistic Regression

   Brown CS142, Spring 2018
"""
import random

import numpy as np


class NaiveBayes(object):
    """ Bernoulli Naive Bayes model

    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes classifer with n_classes. """
        self.n_classes = n_classes
        self.table = np.full((784, self.n_classes), 0.01)
        self.count_class = np.zeros(n_classes)
        # You are free to add more fields here.

    def train(self, data):
        """ Trains the model, using maximum likelihood estimation.

        @params:
            data: the training data as a namedtuple with two fields: inputs and labels
        @return:
            None
        """
        # TODO
        inputs = data.inputs
        labels = data.labels

        for idx, d in enumerate(inputs):
            self.table[:,labels[idx]] += d
            self.count_class[labels[idx]] += 1
        table_sum = np.sum(self.table, axis=1)

        for i in range(784):
            self.table[i,:] /= self.count_class
        self.count_class /= np.sum(self.count_class)




    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.
        @params:
            inputs: a NumPy array containing inputs
        @return:
            a numpy array of predictions
        """
        #TODO
        predict = []
        for idx, ip in enumerate(inputs):
            score = np.zeros(self.n_classes)
            for label in range(self.n_classes):
                for idx2, feat in enumerate(ip):
                    if feat:
                        score[label] += np.log(self.table[idx2,label])
                    else:
                        score[label] += np.log(1-self.table[idx2, label])
                score[label] += np.log(self.count_class[label])

            predict.append(np.argmax(np.array(score)))

        return np.array(predict)

    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        #TODO
        prediction = self.predict(data.inputs)
        correct = np.sum(prediction == data.labels)
        return float(correct)/len(data.labels)

class LogisticRegression(object):
    """ Multinomial Linear Regression

    @attrs:
        weights: a parameter of the model
        alpha: the step size in gradient descent
        n_features: the number of features
        n_classes: the number of classes
    """
    def __init__(self, n_features, n_classes):
        """ Initializes a LogisticRegression classifer. """
        self.alpha = 0.005  # tune this parameter
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = np.ones((n_features, n_classes))

    def train(self, data):
        """ Trains the model, using stochastic gradient descent

        @params:
            data: a namedtuple including training data and its information
        @return:
            None
        """
        diff = 10000
        iteration = 0
        index_array = range(len(data.inputs))
        old_weights = np.ones((self.n_features, self.n_classes))
        l =[]
        l.append(1-self.accuracy(data))
        # while np.allclose(old_weights, self.weights, rtol=1e-01, atol=1e-02, equal_nan=False) is False: ## repeat until convergence
            ##shuffle dataset
        while diff > 0.07 and iteration < 15:
            random.shuffle(index_array)
            iteration+=1
            labels = data[1]
            inputs = data[0]
            old_weights = self.weights
            for idx in index_array:
                loss = np.dot(inputs[idx],self.weights)
                p = self._softmax(loss)
                gradient = np.zeros(self.n_classes)
                gradient = [p[i] -1 if i == labels[idx] else p[i] for i in range(self.n_classes)]

                grad = np.outer(inputs[idx], gradient)
                self.weights = self.weights - self.alpha * grad


            # diff = np.sum(np.sum(np.absolute(old_weights-self.weights)))/ float(np.sum(len(x) for x in self.weights))
            diff = np.absolute(old_weights - self.weights).max()
        return l

        #TODO

    def predict(self, inputs):
        """ Compute predictions based on the learned parameters

        @params:
            data: the training data as a namedtuple with two fields: inputs and labels
        @return:
            None
        """
        prediction = []
        for ip in inputs:
            l = np.dot(ip,self.weights)
            prediction.append(np.argmax(self._softmax(l)))
        return prediction
        #TODO

    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """

        prediction = self.predict(data.inputs)
        correct = np.sum(prediction == data.labels)
        return float(correct)/len(data.labels)


        #TODO

    def _softmax(self, x):
        """ apply softmax to an array

        @params:
            x: the original array
        @return:
            an array with softmax applied elementwise.
        """
        e = np.exp(x - np.max(x))
        return e / np.sum(e)
