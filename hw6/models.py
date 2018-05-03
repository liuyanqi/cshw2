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
        self.num_data = len(X)
        self.b_jy = np.random.uniform(0, 1, (self.num_unique_Y, self.num_hidden))
        self.b_ij = np.random.uniform(0, 1, (self.num_unique_Y, self.num_feature, self.num_hidden))

        for idx, y in enumerate(unique_Y):
            X_ = X[Y == y]
            self.priors[idx] = float(len(X_))/ len(X)

            bjy, bij = self._em_algorithm(X_, self.num_hidden, max_iters, eps)
            self.parameters[y] = (bij, bjy)
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
        bjy = np.random.uniform(0,1, self.num_hidden)
        bij = np.random.uniform(0,1, (self.num_hidden, self.num_feature))
        diff = 10000
        diff2 = 10000
        itr = 0

        while itr < max_iters and (diff > eps and diff2 > eps):
            print(itr, diff, diff2)
            itr += 1
            Q_new = self._e_step(X, self.num_hidden, bjy, bij)
            bij_new, bjy_new = self._m_step(X, self.num_hidden, Q_new)

            diff = np.absolute(bij_new - bij).max() 
            diff2 = np.absolute(bjy_new - bjy).max()

            bjy = bjy_new
            bij = bij_new

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
        Q_t_new = np.zeros((self.num_data, self.num_hidden))
        
        for idx, x in enumerate(X):
            for j in range(self.num_hidden):
                temp = 1
                temp_log = 0
                for i in range(self.num_feature):
                   temp = temp * np.power(bij[j,i], x[i]) * np.power((1-bij[j,i]), (1-x[i]))
                   # log-space computation:
                   # temp_log +=  x[i] * np.log(bij[j,i]) + (1 - x[i]) * np.log(1 - bij[j,i])
                # Q_new_log[j] = temp_log + np.log(bjy[j])
                Q_new[j] = bjy[j] * temp


            Q_new_log = np.log(Q_new)
            Q_new = np.log(Q_new) - (np.amax(Q_new_log) + np.log(np.sum(np.exp(Q_new_log - np.amax(Q_new_log)))))
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

        for i in range(self.num_feature):
            sum_ = 0
            for j in range(len(X)):
                sum_ += probs[j] * X[j,i]

            b_ij[:,i] = sum_ / sumation

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
        print("here")
        for x in range(X):
            prediction = np.zeros(self.num_unique_Y)
            for y in range(self.num_unique_Y):
                temp2 = np.zeros(self.num_hidden)
                for j in range(self.num_hidden):
                    temp =1
                    for i in range(self.num_feature):
                        temp = temp * self.b_ij[y, i, j]

                    temp2[j] =  self.b_jy[y, j] * temp
                # prediction[y] = self.priors[y] * np.sum(temp2)
                #compute in log space
                temp2_log = np.log(temp2)
                prediction[y] = np.log(self.priors[y]) + (np.amax(temp2_log) + np.log(np.sum(np.exp(temp2_log - np.amax(temp2_log)))))

            pre.append(np.argmax(prediction))

        print(pre)
        return np.array(pre)
        

    def accuracy(self, X, Y):
        '''
            Computes the accuracy of classifier on some data (X, Y). The model
            should already be trained before calling accuracy.

            :param X 2D Numpy array where each row, X[i, :] contains a sample. Each sample is a binary feature vector.
            :param Y 1D Numpy array where each element Y[i] is the corresponding label for X[i, :]
            :return A float between 0-1 indicating the fraction of correct predictions.
        '''
        return np.mean(self.predict(X) == Y)
