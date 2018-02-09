import numpy as np
import random
import copy
import math

def train_error(dataset):
    '''
        TODO:
        Calculate the train error of the subdataset and return it.
        For a dataset with two classes:
        C(p) = min{p, 1-p}
    '''
    predict_true = len(dataset[dataset[:,0] ==0])/float(len(dataset))
    return np.min(predict_true, 1-predict_true)
     
    

def entropy(dataset):
    '''
        TODO:
        Calculate the entropy of the subdataset and return it.
        This function is used to calculate the entropy for a dataset with 2 classes.
        Mathematically, this function return:
        C(p) = -p*log(p) - (1-p)log(1-p)
    '''
    if len(dataset) == 0:
        return 0
    predict_true = len(dataset[dataset[:,0]==0])/float(len(dataset))
    entropy = -predict_true*np.log(predict_true) - (1-predict_true)*np.log(1-predict_true)
    return entropy

def gini_index(dataset):
    '''
        TODO:
        Calculate the gini index of the subdataset and return it.
        For dataset with 2 classes:
        C(p) = 2*p*(1-p)
    '''
    if len(dataset) == 0:
        return 0

    predict_true = len(dataset[dataset[:,0]==0])/float(len(dataset))

    return 2 * predict_true * (1-predict_true)



class Node:
    '''
    Helper to construct the tree structure.
    '''
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1, info={}):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on ##QUESTION: which attribute?
        self.isleaf = isleaf #WHERE DO I ASSIGN DIS
        self.label = label
        self.info = info


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0]))) 

        self._split_recurs(self.root, data, indices)

        # Pruning
        if not (validation_data is None):
            self._prune_recurs(self.root, validation_data)

    def predict(self, features):
        '''
        Helper function to predict the label given a row of features.
        You do not need to modify this.
        '''
        return self._predict_recurs(self.root, features)

    def accuracy(self, data):
        '''
        Helper function to calculate the accuracy on the given data.
        You do not need to modify this.
        '''
        return 1 - self.loss(data)

    def loss(self, data):
        '''
        Helper function to calculate the loss on the given data.
        You do not need to modify this.
        '''
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt/len(data)

    def _predict_recurs(self, node, row):
        '''
        Helper function to predict the label given a row of features.
        Traverse the tree until leaves to get the label.
        You do not need to modify this.
        '''
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)


    def _prune_recurs(self, node, validation_data):
        '''
        TODO:
        Prune the tree bottom up recursively. Nothing needs to be returned.
        Do not prune if the node is a leaf.
        Do not prune if the node is non-leaf and has at least one non-leaf child.
        Prune if deleting the node could reduce loss on the validation data.
        '''
        pass



    def _is_terminal(self, node, data, indices):
        '''
        TODO:
        Helper function to determine whether the node should stop splitting.
        Stop the recursion:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the nodex exceede the maximum depth.
        Return:
            - A boolean, True indicating the current node should be a leaf.
            - A label, indicating the label of the leaf (-1 if False)
        '''
        data = np.array(data)
        major_label = int(len(data[data[:,0]==0]) < len(data[data[:,0]==1]))

        if len(data[0])==0:
            print("dataset empty")
            return (True, 0)
        elif node.depth > self.max_depth:
            print("dataset maxdepth")
            return (True, major_label)
        elif not indices :
            print("no more indices")
            return (True, major_label)
        elif len(data[data[:,0]==data[0][0]]) == len(data):
            print("all data same class")
            return (True, data[0][0])
        else:
            return (False, 0)
            


    def _split_recurs(self, node, rows, indices):
        '''
        TODO:
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.
        First use _is_terminal() to check if the node needs to be splitted.
        Then select the column that has the maximum infomation gain to split on.
        Also store the label predicted for this node.
        Then split the data based on whether satisfying the selected column.
        The node should not store data, but the data is recursively passed to the children.
        '''
        #node: parent node
        #rows: all the data left

        ## QUESTION: what's the label of the intermidiate node
        terminal, label = self._is_terminal(node, rows, indices)
        if terminal:
            print("terminal")
            node.label = label
            node.leaf = True
            node.info = {"cost": 0, "data_size": len(rows)}

            return indices
        else:
            print("split")
            rows = np.array(rows)
            max_gain = 0
            max_ind = 0
            for ind in indices:
                gain = self._calc_gain(rows, ind, self.gain_function)
                if gain > max_gain:
                    max_gain = gain
                    max_ind = ind

            print(max_ind, max_gain)
            node.index_split_on = max_ind
            node.info = {"cost": max_gain, "data_size": len(rows)}
            major_label = int(len(rows[rows[:,0]==0]) < len(rows[rows[:,0]==1]))
            node.label = major_label


            indices.remove(max_ind)
            print(indices)
            node.left = Node()
            node.left.depth = node.depth + 1
            indices = self._split_recurs(node.left, rows[rows[:,0]==0], indices)
            node.right = Node()
            node.right.depth = node.depth + 1
            self._split_recurs(node.right, rows[rows[:,0]==1], indices)




    def _calc_gain(self, data, split_index, gain_function):
        '''
        TODO:
        Calculate the gain of the proposed splitting and return it.
        Gain = C(P[y=1]) - (P[x_i=True] * C(P[y=1|x_i=True]) + P[x_i=False]C(P[y=1|x_i=False)])
        Here the C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and gini functions.
        '''
        if len(data) ==0:
            return 0
        prob_x_true = len(data[data[:,split_index]==1])/ float(len(data))
        prob_x_false = 1 - prob_x_true
        Gain = gain_function(data) - prob_x_true * gain_function(data[data[:,split_index]==1]) + prob_x_false * gain_function(data[data[:,split_index]==0])
        return Gain

    def print_tree(self):
        '''
        Helper function for tree_visualization.
        Only effective with very shallow trees.
        You do not need to modify this.
        '''
        temp = []
        output = []
        print('---START PRINT TREE---')
        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = %d; cost = %f; sample size = %d' % (node.index_split_on, node.info['cost'], node.info['data_size'])
            left = indent + 'T -> '+ print_subtree(node.left, indent + '\t\t')
            right = indent + 'F -> '+ print_subtree(node.right, indent + '\t\t')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')




    def loss_plot_vec(self, data):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec)/len(data)



    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        labels = [row[0] for row in rows]
        print(labels)
        print(node.label)
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct


        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
