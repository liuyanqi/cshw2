import numpy as np
from qp import solve_QP
import matplotlib.pyplot as plt
# from sympy.solvers import solver
# from sympy import Symbol 

def linear_kernel(xj, xk):
    """
    Kernel Function, linear kernel (ie: regular dot product)

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :return: float32
    """
    #TODO
    return np.dot(xj, xk)

def rbf_kernel(xj, xk, gamma = 0.1):
    """
    Kernel Function, radial basis function kernel or gaussian kernel

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :param gamma: parameter of the RBF kernel.
    :return: float32
    """
    # TODO
    return np.exp(-gamma*np.linalg.norm(xj-xk)*np.linalg.norm(xj-xk))

def polynomial_kernel(xj, xk, c = 1, d = 2):
    """
    Kernel Function, polynomial kernel

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :param c: mean of the polynomial kernel (np array)
    :param d: exponent of the polynomial (np array)
    :return: float32
    """
    #TODO
    return np.power(np.dot(xj, xk)+c, d)

class SVM(object):

    def __init__(self, kernel_func=linear_kernel, lambda_param=.1):
        self.kernel_func = kernel_func
        self.lambda_param = lambda_param

    def train(self, inputs, labels):
        """
        train the model with the input data (inputs and labels),
        find the coefficients and constaints for the quadratic program and
        calculate the alphas

        :param inputs: inputs of data, a numpy array
        :param labels: labels of data, a numpy array
        :return: None
        """
        self.train_inputs = inputs
        self.train_labels = labels
        Q, c = self._objective_function()
        A, b = self._inequality_constraints()
        E, d = self._equality_constraints()
        # print("A: ", A)
        # print("b: ", b)
        # print("Q: ", Q)
        # TODO: Uncomment the next line when you have implemented _objective_function(),
        # _inequality_constraints() and _equality_constraints().
        self.alphas = solve_QP(Q, c, A, b, E, d)

        alphas_idx = 0
        for idx in range(len(self.train_labels)):
            if  not np.isclose(self.alphas[idx], 0, atol=1e-3) and not np.isclose(self.alphas[idx],(1.0/(2*len(self.train_labels)*self.lambda_param)), atol=1e-3):
                alphas_idx = idx
                break

        self.alphas_idx = alphas_idx
        sum_b = 0
        for idx, x_j in enumerate(self.train_inputs):
            sum_b += self.alphas[idx]*(2*self.train_labels[idx]-1)*self.kernel_func(x_j, self.train_inputs[alphas_idx])
        b = sum_b - (2* self.train_labels[alphas_idx] -1)
        self.b = b
        # print(self.b)



        #TODO: Given the alphas computed by the quadratic solver, compute the bias

        


    def _objective_function(self):
        """
        Generate the coefficients on the variables in the objective function for the
        SVM quadratic program.

        Recall the objective function is:
        minimize (1/2)x^T Q x + c^T x

        For specifics on the values for Q and c, see the objective function in the handout.

        :return: two numpy arrays, Q and c which fully specify the objective function.
        """

        #TODO
        c = -np.ones(len(self.train_labels))
        Q = np.zeros((len(self.train_labels), len(self.train_labels)))

        for i in range(len(self.train_labels)):
            for j in range(len(self.train_labels)):
                Q[i,j] = 0.5*(2*self.train_labels[i]-1)*(2*self.train_labels[j]-1)*self.kernel_func(self.train_inputs[i], self.train_inputs[j])


        return Q, c

    def _equality_constraints(self):
        """
        Generate the equality constraints for the SVM quadratic program. The
        constraints will be enforced so that Ex = d.

        For specifics on the values for E and d, see the constraints in the handout

        :return: two numpy arrays, E, the coefficients, and d, the values
        """

        #TODO
        E = (2*self.train_labels-1).reshape((1, len(self.train_labels)))
        d = np.zeros(1)
        return E, d

    def _inequality_constraints(self):
        """
        Generate the inequality constraints for the SVM quadratic program. The
        constraints will be enforced so that Ax <= b.

        For specifics on the values of A and b, see the constraints in the handout

        :return: two numpy arrays, A, the coefficients, and b, the values
        """
        #TODO
        a1 = -1
        a2 = 1
        b1 = 0
        b2 = 1.0/(2*len(self.train_labels)*self.lambda_param)

        A = np.zeros((2*len(self.train_labels), len(self.train_labels)))
        b = np.zeros(2*len(self.train_labels))
        for i in range(0, len(self.train_labels)):
            A[2*i,i] = a1
            A[2*i+1,i] = a2
            b[2*i] = b1
            b[2*i+1] = b2

        return A, b

    def predict(self, inputs):
        """
        Generate predictions given input.

        :param input: 2D Numpy array. Each row is a vector for which we output a prediction.
        :return: A 1D numpy array of predictions.
        """

        #TODO
        predict = []
        for ip in inputs:
            c = 0
            for i in range(len(self.train_labels)):
                c+=self.alphas[i]*(2*self.train_labels[i]-1)*self.kernel_func(self.train_inputs[i], ip)
            c = c - self.b
            if c > 0:
                predict.append(1)
            else:
                predict.append(0)
        return predict


    def accuracy(self, inputs, labels):
        """
        Calculate the accuracy of the classifer given inputs and their true labels.

        :param inputs: 2D Numpy array which we are testing calculating the accuracy of.
        :param labels: 1D Numpy array with the inputs corresponding true labels.
        :return: A float indicating the accuracy (between 0.0 and 1.0)
        """

        #TODO
        prediction = self.predict(inputs)
        correct = 0

        for i in range(len(prediction)):
            if(prediction[i] ==labels[i]):
                correct += 1
        return correct/float(len(prediction))
