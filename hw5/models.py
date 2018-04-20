import numpy as np
import random


def l2_loss(Y, predictions):
    '''
        Computes L2 loss (sum squared loss) between true values, Y, and predictions.

        :param Y A 1D Numpy array with real values (float64)
        :param predictions A 1D Numpy array of the same size of Y
        :return L2 loss using predictions for Y.
    '''
    # norm = np.linalg.norm(Y-predictions)
    # return np.power(norm, 2)
    return np.sum((Y - predictions) ** 2)
    

def sigmoid(x):
    '''
        Sigmoid function f(x) =  1/(1 + exp(-x))
        :param x A scalar or Numpy array
        :return Sigmoid function evaluated at x (applied element-wise if it is an array)
    '''
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + np.exp(0)))

def sigmoid_derivative(x):
    '''
        First derivative of the sigmoid function with respect to x.
        :param x A scalar or Numpy array
        :return Derivative of sigmoid evaluated at x (applied element-wise if it is an array)
    '''
    # TODO
    return sigmoid(x)*(1.0 - sigmoid(x))

class LinearRegression:
    '''
        LinearRegression model that minimizes squared error using matrix inversion.
    '''
    def __init__(self):
        '''
        @attrs:
            weights The weights of the linear regression model.
        '''
        self.weights = None

    def train(self, X, Y):
        '''
        Trains the LinearRegression model by finding the optimal set of weights
        using matrix inversion.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return None
        '''
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)),Y)
        print(self.weights.shape)

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        # TODO
        return np.dot(X, self.weights)

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]

class OneLayerNN:
    '''
        One layer neural network trained with Stocastic Gradient Descent (SGD)
    '''
    def __init__(self):
        '''
        @attrs:
            weights The weights of the neural network model.
        '''
        self.weights = None
        pass

    def train(self, X, Y, learning_rate=0.001, epochs=250, print_loss=True):
        '''
        Trains the OneLayerNN model using SGD.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :param learning_rate The learning rate to use for SGD
        :param epochs The number of times to pass through the dataset
        :param print_loss If True, print the loss after each epoch.
        :return None
        '''
        # TODO
        self.weights = np.zeros(len(X[0]))
        for e in range(epochs):
            for i, x in enumerate(X):
                logits = np.dot(x, self.weights)
                loss = l2_loss(logits, Y[i])
                w_grad = []
                w = 0
                for j in range(len(X[0])):
                    w = (Y[i] - np.dot(X[i], self.weights))* (-X[i,j])
                    w_grad.append(2*w)

                self.weights = self.weights - learning_rate* np.array(w_grad)

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        #TODO
        return np.dot(X, self.weights)

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]


class TwoLayerNN:

    def __init__(self, hidden_size, activation=sigmoid, activation_derivative=sigmoid_derivative):
        '''
        @attrs:
            activation: the activation function applied after the first layer
            activation_derivative: the derivative of the activation function. Used for training.
            hidden_size: The hidden size of the network (an integer)
            output_neurons: The number of outputs of the network
        '''
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.hidden_size = hidden_size

        # In this assignment, we will only use output_neurons = 1.
        self.output_neurons = 1

        # These are the learned parameters for the 2-Layer NN you will implement
        self.hidden_weights = None
        self.hidden_bias = None
        self.output_weights = None
        self.output_bias = None

    def train(self, X, Y, learning_rate=0.01, epochs=1000, print_loss=True):
        '''
        Trains the TwoLayerNN with SGD using Backpropagation.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :param learning_rate The learning rate to use for SGD
        :param epochs The number of times to pass through the dataset
        :param print_loss If True, print the loss after each epoch.
        :return None
        '''
        #TODO
        self.hidden_weights = np.zeros((len(X[0]), self.hidden_size))
        self.hidden_bias = np.zeros(self.hidden_size)
        self.output_weights = np.zeros(self.hidden_size)
        self.output_bias = np.zeros(1)
        for ep in range(epochs):
            print(ep)
            index = np.arange(len(X))
            np.random.shuffle(index)
            if print_loss:
                print(self.average_loss(X, Y))
            for i in index:

                logits = np.dot(X[i], self.hidden_weights) + self.hidden_bias
                h = sigmoid(logits)
                z = np.dot(h, self.output_weights) + self.output_bias

                obias_grad = 2*(z - Y[i])

                vi_grad = np.zeros(self.hidden_size)
                b1i_grad = np.zeros(self.hidden_size)
                wij_grad = np.zeros(self.hidden_weights.shape)

                for j in range(self.hidden_size):
                    vi_grad[j] = 2* (z - Y[i]) * h[j]

                    grad = 2* (z - Y[i]) * self.output_weights[j]
                    grad = grad *sigmoid_derivative(np.dot(X[i], self.hidden_weights[:,j]) + self.hidden_bias[j])
                    b1i_grad[j] = grad

                    for k in range(len(X[0])):
                        w_grad = 2*(z - Y[i]) * self.output_weights[j]
                        w_grad = np.dot(w_grad, X[i,k])
                        first = np.dot(X[i], self.hidden_weights[:,j]) + self.hidden_bias[j]
                        w_grad = w_grad * sigmoid_derivative(first)
                        wij_grad[k,j] = w_grad

                
                # update step
                self.hidden_weights = self.hidden_weights - learning_rate * wij_grad
                self.hidden_bias = self.hidden_bias - learning_rate * b1i_grad
                self.output_weights = self.output_weights - learning_rate * vi_grad
                self.output_bias = self.output_bias - learning_rate * obias_grad



    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        #TODO
        h = np.dot(X, self.hidden_weights) + self.hidden_bias
        h = sigmoid(h)
        z = np.dot(h, self.output_weights) + self.output_bias
        return z

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]
