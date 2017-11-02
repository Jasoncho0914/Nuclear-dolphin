from clustering_info_meta import load_graph
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
import os
import numpy as np
import tflearn
import tensorflow as tf
import jsonpickle as jp

import copy

import random

def split_set(s, fraction):
    s1 = set()
    s2 = set()

    for i, x in enumerate(s):
        if not i % fraction == 0:
            s1.add(x)
        else:
            s2.add(x)
    return s1, s2

def create_train_set(dataset, connected_pairs, unconnected_pairs):
    inputs = []
    outputs = []
    for a, b in connected_pairs:
        row = copy.deepcopy(dataset[a])
        row.extend(dataset[b])
        inputs.append(row)
        outputs.append([1])
    for a, b in unconnected_pairs:
        row = copy.deepcopy(dataset[a])
        row.extend(dataset[b])
        inputs.append(row)
        outputs.append([-1])
    return np.asarray(inputs), np.asarray(outputs)



def main():
    data_matrix = np.ndarray.tolist(np.loadtxt(open("Scripts/Results/pca30.csv", "rb"), delimiter=","))

    graph = load_graph(True)
    graph_keys = [x for x in graph.keys()]

    edge_count = 0
    for key in graph_keys:
        for _ in graph[key]:
            edge_count += 1
    edge_count /= 2 # because there are duplicates

    print("edge count {}".format(edge_count))

    scores = []
    train_set_size = 500000
    try:
        while train_set_size < edge_count / 2:
            print("Training for training set size of {}".format(train_set_size))

            print("collect train set ... ")
            # get 'train_set_size' random datapoints for connected datapoints
            train_connected_pairs = set()
            while len(train_connected_pairs) < train_set_size:
                a = random.choice(graph_keys)
                b = random.choice(graph[a])
                pair = (a, b,)
                if not pair in train_connected_pairs:
                    train_connected_pairs.add(pair)

            # same for unconnected datapoints
            train_unconnected_pairs = set()
            while len(train_unconnected_pairs) < train_set_size * 4:
                a = random.randint(0, 6000-1)
                b = random.randint(0, 6000-1)
                tup = (a,b,)
                if not (a in graph and b in graph[a]):
                    train_unconnected_pairs.add(tup)

            # partition into train and validation sets
            train_connected_pairs, dev_connected_pairs = split_set(train_connected_pairs, 5)
            train_unconnected_pairs, dev_unconnected_pairs = split_set(train_unconnected_pairs, 5)

            # now assemble into training matrix and validation matrix
            train_inputs, train_outputs = create_train_set(data_matrix, train_connected_pairs, train_unconnected_pairs)
            dev_inputs, dev_outputs = create_train_set(data_matrix, dev_connected_pairs, dev_unconnected_pairs)

            #train the model
            print("Train Model ... ")
            # model = SVR(kernel='rbf', C=1e3, gamma=0.1) # LinearRegression()
            # model.fit(train_inputs, train_outputs)
            model = MLPClassifier(hidden_layer_sizes=(25, 25, 25, 25,), early_stopping=True)
            model.fit(train_inputs, train_outputs)

            # tf.reset_default_graph() # necessary for making new nnets without cryptic error
            # net = tflearn.input_data(shape=[None, 60])
            # net = tflearn.fully_connected(net, 20, activation='sigmoid')
            # #tflearn.losses.L2(net)
            # net = tflearn.fully_connected(net, 20, activation='sigmoid')
            # #tflearn.losses.L2(net)
            # net = tflearn.fully_connected(net, 20, activation='sigmoid')
            # #tflearn.losses.L2(net)
            # net = tflearn.fully_connected(net, 1, activation='sigmoid')
            # net = tflearn.regression(net, optimizer='adam') # loss='categorical_crossentropy'
            # model = tflearn.DNN(net)


            model.fit(train_inputs, train_outputs)

            # test the model
            print("Score Model ... ")
            false_positives = 0
            false_negatives = 0
            right = 0
            total = 0
            guesses = np.ndarray.tolist(model.predict(dev_inputs))
            correct_answers = np.ndarray.tolist(dev_outputs)
            for guess, correct, in zip(guesses, correct_answers):
                # guess = guess[0]
                correct = correct[0]
                # if abs(guess) > .25: # only sample accuracy in cases of high confidence
                # print("{}, {}".format(guess, correct))
                if (guess <= 0 and correct <= 0) or guess >= 0 and correct >= 0:
                    right += 1
                else:
                    if correct == -1:
                        false_positives += 1
                    else:
                        false_negatives += 1
                total += 1
            print()
            print("Percentage correct: {} of {}".format(round(right * 100 / total, 2), total))
            print("False positives: {}".format(round(false_positives*100 / total, 2)))
            print("False negatives: {}".format(round(false_negatives * 100 / total, 2)))
            scores.append( ( round(right * 100 / total, 2), train_set_size,))

            print("Writing to file...")
            f = open('models/predict_connection_{}.json'.format(train_set_size), 'w')
            f.write(jp.encode(model))
            f.close()

            train_set_size *= 2
    except KeyboardInterrupt:
        print("Scores:")
        print(scores)
        quit()
    print(scores)
    print("DONE")






if __name__ == '__main__':
    main()