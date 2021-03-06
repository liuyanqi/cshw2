import numpy as np


class EMNaiveBayesClassifier:

    def __init__(self, num_hidden):
        '''
        @attrs:
            num_hidden  The number of hidden states per class. (An integer)
            priors The estimated prior distribution over the classes, P(Y) (A Numpy Array)
            parameters The estimated parameters of the model. A Python dictionary from class to the parameters
                conditional on that class. More specifically, the dictionary at parameters[Y] should store
                - bjy: b^jy = P(h^j | Y) for each j = 1 ... k
                - bij: b^ij = P(x_i | h^j, Y)) for each i, for each j = 1 ... k
        '''
        self.num_hidden = num_hidden
        self.priors = None
        self.parameters = None
        pass

    def train(self, X, Y, max_iters=10, eps=1e-4):
        '''
            Trains the model using X, Y. More specifically, it learns the parameters of
            the model. It must learn:
                - b^y = P(y) (via MLE)
                - b^jy = P(h^j | Y)  (via EM algorithm)
                - b^ij = P(x_i | h^j, Y) (via EM algorithm)

            Before running the EM algorithm, you should partition the dataset based on the labels Y. Then
            run the EM algorithm on each of the subsets of data.

            :param X 2D Numpy array where each row, X[i, :] contains a sample. Each sample is a binary feature vector.
            :param Y 1D Numpy array where each element Y[i] is the corresponding label for X[i, :]
            :max_iters The maxmium number of iterations to run the EM-algorithm. One
                iteration consists of both the E-step and the M-step.
            :eps Used for convergence test. If the maximum change in any of the parameters is
                eps, then we should stop running the algorithm.
            :return None
        '''
        # TODO

        unique_Y = np.unique(Y)
        self.num_unique_Y = len(unique_Y)
        self.priors = np.zeros(self.num_unique_Y)
        self.parameters = {}

        self.num_feature = len(X[0])
        self.b_jy = np.random.uniform(0, 1, (self.num_unique_Y, self.num_hidden))
        self.b_ij = np.random.uniform(0, 1, (self.num_unique_Y, self.num_hidden, self.num_feature))

        for idx, y in enumerate(unique_Y):
            X_ = X[Y == y]
            self.priors[idx] = float(len(X_))/ len(X)

            bjy, bij = self._em_algorithm(X_, self.num_hidden, max_iters, eps)
            # self.parameters[y] = (bij, bjy)
            self.parameters[y] = {}
            self.parameters[y]['bij'] = bij
            self.parameters[y]['bjy'] = bjy
            self.b_jy[idx] = bjy
            self.b_ij[idx] = bij



    def _em_algorithm(self, X, num_hidden, max_iters, eps):
        '''
            EM Algorithm to learn parameters of a Naive Bayes model.

            :param X A 2D Numpy array containing the inputs.
            :max_iters The maxmium number of iterations to run the EM-algorithm. One
                iteration consists of both the E-step and the M-step.
            :eps Used for convergence test. If the maximum change in any of the parameters is
                eps, then we should stop running the algorithm.
            :return the learned parameters as a tuple (b^ij,b^jy)
        '''
        # TODO
        bjy = np.random.uniform(0.001,1, self.num_hidden)
        bij = np.random.uniform(0.001,1, (self.num_hidden, self.num_feature))
        diff = 10000
        diff2 = 10000
        itr = 0

        while itr < max_iters and (diff > eps and diff2 > eps):
            Q_new = self._e_step(X, self.num_hidden, bjy, bij)
            bij_new, bjy_new = self._m_step(X, self.num_hidden, Q_new)

            diff = np.absolute(bij_new - bij).max() 
            diff2 = np.absolute(bjy_new - bjy).max()

            bjy = bjy_new
            bij = bij_new

            itr += 1

        return(bjy, bij)


    def _e_step(self, X, num_hidden, bjy, bij):
        '''
            The E-step of the EM algorithm. Returns Q(t+1) = P(h^j | x, y, theta)
            See the handout for details.

            :param X The inputs to the EM-algorthm. A 2D Numpy array.
            :param num_hidden The number of latent states per class (k in the handout)
            :param bjy at the current iteration (b^jy = P(h^j | y))
            :param bij at the current iteration (b^ij = P(x_i | h^j, y))
            :return Q(t+1)
        '''
        # TODO
        Q_new = np.zeros(self.num_hidden)
        Q_new_log = np.zeros(self.num_hidden)
        Q_t_new = np.zeros((len(X), self.num_hidden))
        
        for idx, x in enumerate(X):
            for j in range(self.num_hidden):
                temp_log = 0
                # print(bij[j,:])
                temp_log = np.dot(x, np.log(bij[j,:]))
                temp_log += np.dot((1-x), np.log(1 - bij[j,:]))
                # for i in range(self.num_feature):
                #    # temp = temp * np.power(bij[j,i], x[i]) * np.power((1-bij[j,i]), (1-x[i]))
                #    # log-space computation:
                #    temp_log +=  x[i] * np.log(bij[j,i]) + (1 - x[i]) * np.log(1 - bij[j,i])
                Q_new_log[j] = temp_log + np.log(bjy[j])
                # Q_new[j] = bjy[j] * temp


            # Q_new_log = np.log(Q_new)
            Q_new = Q_new_log - (np.amax(Q_new_log) + np.log(np.sum(np.exp(Q_new_log - np.amax(Q_new_log)))))
            Q_t_new[idx] = np.exp(Q_new)

        return Q_t_new




    def _m_step(self, X, num_hidden, probs):
        '''
            The M-step of the EM algorithm. Returns the next update to the parameters,
            theta(t+1).

            :param X The inputs to the EM-algorthm. A 2D Numpy array.
            :param num_hidden The number of latent states per class (k in the handout)
            :param probs (Q(t))
            :return theta(t+1) as a tuple (b^ij,b^jy)
        '''
        # TODO

        S_y = len(X)
        sumation = np.sum(probs, axis=0)
        b_jy = sumation/S_y

        b_ij = np.zeros((self.num_hidden, self.num_feature))
        sum_ =np.dot(np.transpose(probs),X)

        b_ij = np.transpose(np.transpose(sum_) / sumation)
        b_ij[b_ij == 0] += 1e-10
        b_ij[b_ij >  (1-1e-10)] = b_ij[b_ij  > (1-1e-10)] - 1e-10
        # print(b_ij)
        
        # for i in range(self.num_feature):
        #     sum_ = np.dot(np.transpose(probs), X[:,i])
            # for j in range(len(X)):
            #     print(probs[j].shape)

            #     sum_ += probs[j] * X[j,i]

            # b_ij[:,i] = sum_ / sumation

        return(b_ij, b_jy)


    def predict(self, X):
        '''
        Returns predictions for the vectors X. For some input vector x,
        the classifier should output y such that y = argmax P(y | x),
        where P(y | x) is approximated using the learned parameters of the model.

        :param X 2D Numpy array. Each row contains an example.
        :return A 1D Numpy array where each element contains a prediction 0 or 1.
        '''
        # TODO
        pre = []
        # for x in X:
        #     prediction = np.zeros(self.num_unique_Y)
        #     for y in range(self.num_unique_Y):
        #         temp2 = np.zeros(self.num_hidden)
        #         for j in range(self.num_hidden):

        #             # bi = self.b_ij[y, j, :]
        #             # bi[bi==0] += 1e-10
        #             # bi[bi==1] -= 1e-10
        #             # # print(bi[bi==1])
        #             # bi = [1 - bi[idx] for idx in range(len(x)) if x[idx]==0]
        #             # temp_log = np.sum(np.log(bi))
        #             # temp_log += np.log(self.b_jy[y,j])
        #             # temp2[j] = temp_log

        #             temp = 0
        #             for i in range(self.num_feature):
        #                 if x[i] == 1:
        #                     temp = temp + np.log(self.b_ij[y,j,i])
        #                 else:
        #                     temp = temp + np.log(1-self.b_ij[y,j,i]) 
        #             temp2[j] =  np.log(self.b_jy[y, j]) + temp

        #         #compute in log space
        #         prediction[y] = np.log(self.priors[y]) + (np.amax(temp2) + np.log(np.sum(np.exp(temp2 - np.amax(temp2)))))
        #     pre.append(np.argmax(prediction))
        # return np.array(pre)
        prediction = np.zeros((len(X), self.num_unique_Y))
        for y, dict_ in self.parameters.items():
            b_ij = dict_['bij']
            b_jy = dict_['bjy']
            for i, x in enumerate(X):
                pre = np.exp(np.sum(np.log(b_ij[:, X[i, :] == 1]), axis=1) +np.sum(np.log(1 - b_ij[:, X[i, :] == 0]), axis=1) + b_jy)
                pre = np.sum(pre) * self.priors[y]
                prediction[i][y] = pre

        return np.argmax(prediction, axis=1)


    def accuracy(self, X, Y):
        '''
            Computes the accuracy of classifier on some data (X, Y). The model
            should already be trained before calling accuracy.

            :param X 2D Numpy array where each row, X[i, :] contains a sample. Each sample is a binary feature vector.
            :param Y 1D Numpy array where each element Y[i] is the corresponding label for X[i, :]
            :return A float between 0-1 indicating the fraction of correct predictions.
        '''
        # prediction = self.predict(X)
        return np.mean(self.predict(X) == Y)
