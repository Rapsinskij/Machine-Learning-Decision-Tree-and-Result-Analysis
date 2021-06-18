##############
# Name:Jason Rapsinski
# email:jrapsins@data.cs.purdue.edu
# Date:10/18/2020

import numpy as np
import sys
import os
import pandas as pd
import math
import copy

def entropy(freqs):
    all_freq = sum(freqs)
    entropy = 0
    for fq in freqs:
        prob = fq * 1.0 / all_freq
        if abs(prob) > 1e-8:
            entropy += -prob * np.log2(prob)
    return entropy

def infor_gain(before_split_freqs, after_split_freqs):
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    for freq in after_split_freqs:
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain


class Node(object):
    def __init__(self, pre, l, r, attr, thresh, result, elements):
        self.parent = pre
        self.left_subtree = l
        self.right_subtree = r
        self.attribute = attr
        self.threshold = thresh
        self.survived = result
        self.element_count = elements


class Tree(object):
	def __init__(self, ___):
		pass

def ID3(train_data, train_labels, prev_attr):
    highest_gain = -1
    survive_class = train_labels.survived.mode().get(0)
    before_split = [(train_labels.survived == 0).sum(), (train_labels.survived == 1).sum()]
    combined_list = train_data.copy()
    combined_list['survived'] = train_labels
    # iterate through attributes
    for att in train_data:
        # iterate through attribute values
        NaN_replacement = getattr(combined_list, att).median()
        combined_list[att] = combined_list[att].fillna(NaN_replacement)
        for unique_val in getattr(train_data,att).unique():
            if not math.isnan(unique_val) and att != prev_attr:
                # split data across threshold
                lesser_eq = combined_list.loc[getattr(combined_list,att) <= unique_val]
                greater = combined_list.loc[getattr(combined_list,att) > unique_val]
                after_split = [[(lesser_eq.survived == 0).sum(), (lesser_eq.survived == 1).sum()], [(greater.survived==0).sum(), (greater.survived==1).sum()]]
                # calculate information gain
                curr_gain = infor_gain(before_split, after_split)

                #keep attribute and threshold with highest information gain
                if curr_gain > highest_gain:
                    highest_gain = curr_gain
                    the_chosen_attribute = att
                    the_chosen_threshold = unique_val
                    left_part_train_label = pd.DataFrame(lesser_eq.pop('survived'))
                    left_part_train_data = lesser_eq.copy()
                    right_part_train_label = pd.DataFrame(greater.pop('survived'))
                    right_part_train_data = greater.copy()

    # set attribute and threshold for current node
    current_node = Node(None, None, None, the_chosen_attribute, the_chosen_threshold, survive_class, len(train_labels))
    if highest_gain >0:
        # recursively call function on left and right data splits
        left_subtree = ID3(left_part_train_data, left_part_train_label, the_chosen_attribute)
        left_subtree.parent = current_node
        right_subtree = ID3(right_part_train_data, right_part_train_label, the_chosen_attribute)
        right_subtree.parent = current_node
        current_node.left_subtree = left_subtree
        current_node.right_subtree = right_subtree
    return current_node

# same basic framwork as ID3 but with added functionality for max_depth
def ID3_depth(train_data, train_labels, prev_attr, level, max_depth):
    # 1. use a for loop to calculate the infor-gain of every attribute
    highest_gain = -1
    survive_class = train_labels.survived.mode().get(0)
    before_split = [(train_labels.survived == 0).sum(), (train_labels.survived == 1).sum()]
    combined_list = train_data.copy()
    combined_list['survived'] = train_labels
    for att in train_data:
        # 1.1 pick a threshold
        for unique_val in getattr(train_data,att).unique():
            if not math.isnan(unique_val) and att != prev_attr:
                # 1.2 split the data using the threshold
                lesser_eq = combined_list.loc[getattr(combined_list,att) <= unique_val]
                greater = combined_list.loc[getattr(combined_list,att) > unique_val]
                after_split = [[(lesser_eq.survived == 0).sum(), (lesser_eq.survived == 1).sum()], [(greater.survived==0).sum(), (greater.survived==1).sum()]]
                # 1.3 calculate the infor_gain
                curr_gain = infor_gain(before_split, after_split)
                if curr_gain > highest_gain:
                    highest_gain = curr_gain
                    the_chosen_attribute = att
                    the_chosen_threshold = unique_val
                    left_part_train_label = pd.DataFrame(lesser_eq.pop('survived'))
                    left_part_train_data = lesser_eq.copy()
                    right_part_train_label = pd.DataFrame(greater.pop('survived'))
                    right_part_train_data = greater.copy()

    # 2. pick the attribute that achieve the maximum infor-gain
    # 3. build a node to hold the data;
    current_node = Node(None, None, None, the_chosen_attribute, the_chosen_threshold, survive_class, len(train_labels))
    if highest_gain >0:
        # 4. split the data into two parts.
        # 5. call ID3() for the left parts of the data
        # only recursively generates children if tree is not at max depth
        if level < max_depth:
            left_subtree = ID3_depth(left_part_train_data, left_part_train_label, the_chosen_attribute, level+1, max_depth)
            left_subtree.parent = current_node
            # 6. call ID3() for the right parts of the data.
            right_subtree = ID3_depth(right_part_train_data, right_part_train_label, the_chosen_attribute, level+1, max_depth)
            right_subtree.parent = current_node
            current_node.left_subtree = left_subtree
            current_node.right_subtree = right_subtree
    return current_node

# same framework as ID3 but with added functionality for min_split
def ID3_minSplit(train_data, train_labels, prev_attr, min_split):
    # 1. use a for loop to calculate the infor-gain of every attribute
    highest_gain = -1
    survive_class = train_labels.survived.mode().get(0)
    before_split = [(train_labels.survived == 0).sum(), (train_labels.survived == 1).sum()]
    combined_list = train_data.copy()
    combined_list['survived'] = train_labels
    for att in train_data:
        # 1.1 pick a threshold
        for unique_val in getattr(train_data,att).unique():
            if not math.isnan(unique_val) and att != prev_attr:
                # 1.2 split the data using the threshold
                lesser_eq = combined_list.loc[getattr(combined_list,att) <= unique_val]
                greater = combined_list.loc[getattr(combined_list,att) > unique_val]
                after_split = [[(lesser_eq.survived == 0).sum(), (lesser_eq.survived == 1).sum()], [(greater.survived==0).sum(), (greater.survived==1).sum()]]
                # 1.3 calculate the infor_gain
                curr_gain = infor_gain(before_split, after_split)
                if curr_gain > highest_gain:
                    highest_gain = curr_gain
                    the_chosen_attribute = att
                    the_chosen_threshold = unique_val
                    left_part_train_label = pd.DataFrame(lesser_eq.pop('survived'))
                    left_part_train_data = lesser_eq.copy()
                    right_part_train_label = pd.DataFrame(greater.pop('survived'))
                    right_part_train_data = greater.copy()

    # 2. pick the attribute that achieve the maximum infor-gain
    # 3. build a node to hold the data;
    current_node = Node(None, None, None, the_chosen_attribute, the_chosen_threshold, survive_class, len(train_labels))
    if highest_gain >0:
        # 4. split the data into two parts.
        # 5. call ID3() for the left parts of the data
        left_subtree = ID3_minSplit(left_part_train_data, left_part_train_label, the_chosen_attribute, min_split)
            # 6. call ID3() for the right parts of the data.
        right_subtree = ID3_minSplit(right_part_train_data, right_part_train_label, the_chosen_attribute, min_split)
        # only creates children if element count is greater than min_split
        # this should definitely be earlier to avoid unnecessary calculations
        # but I don't have time to fix it and verify that it still works
        if left_subtree.element_count >= min_split and right_subtree.element_count >= min_split:
            left_subtree.parent = current_node
            right_subtree.parent = current_node
            current_node.left_subtree = left_subtree
            current_node.right_subtree = right_subtree
    return current_node

# helper function for pruning
def preorder(root):
    node_list =[]
    if hasattr(root, 'attribute'):
        if hasattr(root.left_subtree, 'attribute') and hasattr(root.right_subtree, 'attribute'):
            node_list.extend(preorder(root.left_subtree))
            node_list.extend(preorder(root.right_subtree))
        node_list.extend([root])
    return node_list
        

# pruning function for vanilla decision tree
def post_pruning(test_data, test_labels, root):
    before_acc = 0
    curr_highest_acc = 0.01
    curr = root
    while curr_highest_acc > before_acc:
        # get accuracy of current tree
        testing = 0.0
        correct = 0.0
        for index, row in test_data.iterrows():
            prediction = get_prediction(curr, row)
            testing=testing+1.0
            actual_results = test_labels.at[index, 'survived']
            if prediction == actual_results:
                correct=correct+1.0
        before_acc = 100*correct/testing
        
        # get list of nodes
        node_list = preorder(curr)
        highest_acc = 0
        best_root = None
        # for each node find accuracy of tree if node had no children
        for node in node_list:
            if hasattr(node, 'attribute') and hasattr(node.left_subtree, 'attribute') and hasattr(node.right_subtree, 'attribute'):
                pruned = copy.deepcopy(node)
                pruned.left_subtree = None
                pruned.right_subtree = None
                while hasattr(pruned.parent, 'attribute'):
                    pruned = pruned.parent

                testing = 0.0
                correct = 0.0
                for index, row in test_data.iterrows():
                    prediction = get_prediction(pruned, row)
                    testing=testing+1.0
                    actual_results = test_labels.at[index, 'survived']
                    if prediction == actual_results:
                        correct=correct+1.0
                after_acc = 100*correct/testing
                
                # keep tree with modified node with highest accuracy
                if after_acc >= curr_highest_acc :
                    curr_highest_acc = after_acc
                    best_root = pruned
        
        curr=best_root

    return curr

#iterates through tree for the given row and predicts outcome
def get_prediction(root, dataset):  
    if hasattr(root.left_subtree, 'attribute') and hasattr(root.right_subtree, 'attribute'):
        att = root.attribute
        val = getattr(dataset, att)
        threshold = root.threshold
        if(val <= threshold):
            survived = get_prediction(root.left_subtree, dataset)
        else:
            survived = get_prediction(root.right_subtree, dataset)
    else:
        survived = root.survived
    return survived

class PCA(object):
    def __init__(self, n_component):
        self.n_component = n_component
    
    #def fit_transform(self, train_data):
        #[TODO] Fit the model with train_data and 
        # apply the dimensionality reduction on train_data.

        
    #def transform(self, test_data):
        #[TODO] Apply dimensionality reduction to test_data.

        
if __name__ == "__main__":
    # parse arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='CS373 Homework2 Decision Tree')
    parser.add_argument('--trainFolder')
    parser.add_argument('--testFolder')
    parser.add_argument('--model')
    parser.add_argument('--depth')
    parser.add_argument('--minSplit')
    parser.add_argument('--crossValidK', type=int, default=5)
    args = parser.parse_args()

    # build decision tree
    # get data from files
    train_file = args.trainFolder
    train_data = pd.read_csv(train_file+".data", delimiter=',', index_col=None, engine='python')
    train_labels = pd.read_csv(train_file+".label", delimiter=',', index_col=None, engine='python')

    test_file = args.testFolder
    test_data = pd.read_csv(test_file+".data", delimiter=',', index_col=None, engine='python')
    test_labels = pd.read_csv(test_file+".label", delimiter=',', index_col=None, engine='python')
    
    cross_validation = args.crossValidK
    validation_set_size = math.ceil(len(train_data)/cross_validation)
    
    best_model = None
    highest_acc = 0
    
    # used to get average accuracies with multiple depths for part2.4
    if args.model=="depthRangeTest":
        depths_range = [2,4,6,8,10,12,14,16]
        for depth in depths_range:
            total_training = 0
            total_validation = 0
            node_count = 0
            for x in range(0, cross_validation):
                start = x*validation_set_size;
                stop = min((x+1)*validation_set_size-1, len(train_data))

                cross_training_data = train_data.copy()
                cross_training_data = cross_training_data.drop(cross_training_data.index[start:stop])
                cross_training_labels = train_labels.copy()
                cross_training_labels = cross_training_labels.drop(cross_training_labels.index[start:stop])
        
                cross_testing_data = train_data[start:stop]
                cross_testing_labels = train_labels[start:stop]
            
                root = ID3_depth(cross_training_data, cross_training_labels, None, 1, depth)
                testing = 0.0
                correct = 0.0
                for index, row in cross_training_data.iterrows():
                    prediction = get_prediction(root, row)
                    testing=testing+1.0
                    actual_results = cross_training_labels.at[index,'survived']
                    if prediction == actual_results:
                        correct=correct+1.0
                training_acc = 100*correct/testing
                total_training += training_acc

                testing = 0.0
                correct = 0.0
                for index, row in cross_testing_data.iterrows():
                    prediction = get_prediction(root, row)
                    testing=testing+1.0
                    actual_results = cross_testing_labels.at[index, 'survived']
                    if prediction == actual_results:
                        correct=correct+1.0
                validation_acc = 100*correct/testing
                total_validation += validation_acc

                node_list = preorder(root)
                node_count += len(node_list)

                if validation_acc > highest_acc:
                    highest_acc = validation_acc
                    best_model = root
            
            total_training = total_training/cross_validation
            total_validation = total_validation/cross_validation
            node_count = node_count/cross_validation
            print("Depth: ", depth, ", Training Accuracy: ", round(total_training,1), " Validation Accuracy: ", round(total_validation,1), ", Average Node Count: ", round(node_count,0))
        sys.exit()

    # used to get average accuracies for different minSplits for part 2.5
    elif args.model=="minSplitRangeTest":
        split_range = [2,4,6,8,10,12,14,16]
        for split_size in split_range:
            total_training = 0
            total_validation = 0
            node_count = 0
            for x in range(0, cross_validation):
                start = x*validation_set_size;
                stop = min((x+1)*validation_set_size-1, len(train_data))

                cross_training_data = train_data.copy()
                cross_training_data = cross_training_data.drop(cross_training_data.index[start:stop])
                cross_training_labels = train_labels.copy()
                cross_training_labels = cross_training_labels.drop(cross_training_labels.index[start:stop])
        
                cross_testing_data = train_data[start:stop]
                cross_testing_labels = train_labels[start:stop]
            
                root = ID3_minSplit(cross_training_data, cross_training_labels, None, split_size)
                testing = 0.0
                correct = 0.0
                for index, row in cross_training_data.iterrows():
                    prediction = get_prediction(root, row)
                    testing=testing+1.0
                    actual_results = cross_training_labels.at[index,'survived']
                    if prediction == actual_results:
                        correct=correct+1.0
                training_acc = 100*correct/testing
                total_training += training_acc

                testing = 0.0
                correct = 0.0
                for index, row in cross_testing_data.iterrows():
                    prediction = get_prediction(root, row)
                    testing=testing+1.0
                    actual_results = cross_testing_labels.at[index, 'survived']
                    if prediction == actual_results:
                        correct=correct+1.0
                validation_acc = 100*correct/testing
                total_validation += validation_acc

                node_list = preorder(root)
                node_count += len(node_list)

                if validation_acc > highest_acc:
                    highest_acc = validation_acc
                    best_model = root
            
            total_training = total_training/cross_validation
            total_validation = total_validation/cross_validation
            node_count = node_count/cross_validation
            print("Minsplit: ", split_size, ", Training Accuracy: ", round(total_training, 1), " Validation Accuracy: ", round(total_validation, 1), ", Average Node Count: ", round(node_count, 0))
        sys.exit()

    # split data into training data with size k-1/k of original and 
    # validation data with size 1/k of original
    for x in range(0, cross_validation):
        start = x*validation_set_size;
        stop = min((x+1)*validation_set_size-1, len(train_data))

        cross_training_data = train_data.copy()
        cross_training_data = cross_training_data.drop(cross_training_data.index[start:stop])
        cross_training_labels = train_labels.copy()
        cross_training_labels = cross_training_labels.drop(cross_training_labels.index[start:stop])
        
        cross_testing_data = train_data[start:stop]
        cross_testing_labels = train_labels[start:stop]

        # builds vanilla decision tree
        if args.model =="vanilla":
            root = ID3(cross_training_data, cross_training_labels, None)

        # builds decision tree with max depth
        elif args.model=="depth":
            max_depth = args.depth
            root = ID3_depth(cross_training_data, cross_training_labels, None, 1, int(max_depth))
        
        # builds decision tree with minSplit
        elif args.model=="minSplit":
            min_split = args.minSplit
            root = ID3_minSplit(cross_training_data, cross_training_labels, None, int(min_split))
        
        # builds vanilla decision tree then prunes it afterwards
        elif args.model=="postPrune":
            root = ID3(cross_training_data, cross_training_labels, None)
            root = post_pruning(cross_testing_data, cross_testing_labels, root)

        # ends program if no recognized model styles
        else:
            print("No valid model styles detected")
            sys.exit()

        # predict on training data and find accuracy
        # probably should have created a helper function to keep code cleaner but this works
        testing = 0.0
        correct = 0.0
        for index, row in cross_training_data.iterrows():
            prediction = get_prediction(root, row)
            testing=testing+1.0
            actual_results = cross_training_labels.at[index,'survived']
            if prediction == actual_results:
                correct=correct+1.0
        training_acc = 100*correct/testing

        # predict on validation data and find accuracy
        testing = 0.0
        correct = 0.0
        for index, row in cross_testing_data.iterrows():
            prediction = get_prediction(root, row)
            testing=testing+1.0
            actual_results = cross_testing_labels.at[index, 'survived']
            if prediction == actual_results:
                correct=correct+1.0
        validation_acc = 100*correct/testing

        # keep the model with highest validation accuracy
        if validation_acc > highest_acc:
            highest_acc = validation_acc
            best_model = root

        print("fold=", x+1, ", train set accuracy=", round(training_acc,1), "%, validation set accuracy=", round(validation_acc,1), "%")

    # use best model to make predictions on test set
    testing = 0.0
    correct = 0.0
    for index, row in test_data.iterrows():
        prediction = get_prediction(best_model, row)
        testing=testing+1.0
        actual_results = test_labels.at[index, 'survived']
        if prediction == actual_results:
            correct=correct+1.0
    test_acc = 100*correct/testing

    print("Test set accuracy=", test_acc, "%")
