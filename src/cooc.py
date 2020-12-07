#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
from collections import Counter

"""
Description

###First Part: 
Open the previously constructed vocab.pkl and retrieve the dictionary as vocab.

###Second Part: 
Will do the following for both train_pos_full.txt and train_neg_full.txt:

- Open the file

- for every line of the file (so for every tweet)

    - Store in a list called tokens the id of each word in the tweet.
      (respective id found in the vocab dictionnary if exists, -1 otherwise)

    - Then get rid of the -1 values.

    ==> At this stage, we have in the list 'tokens' the identifiers of all words
    in the tweet that are in our predefined vocabulary. 

    Now we need build co-occurence matrix. Meaning that we want to count for every word,
    the number of times they co-occure with another word (in a fix Context Window). Here,
    the Context Windown is the length of a tweet.
    The co-occurence matrix is therefore a matrix of size (len(vocab) x len(vocab)), each line
    and column corresponding to a word in the vocabulary (0-th column i the 0-th word is the one with id 0 in the dictionnary)
    The value at the index [i,j] correspond to the number of times i and j occured together.

    So now, for every index t in tokens, we will go over all the index t2 in tokens and increase the count of (t,t2) in the Counter
    by 1.

When all of this is done, we will get three lists:
- data : filled with integers --> the entries of the matrix
- row : filled with ids --> the row indices of the matrix entries
- col : filled with ids --> the column indicies of the matrix entries

Then we will feed these three lists to the coo_matrix function (from scipy.sparse).
coo_matrix[i[k], j[k]] = data[k].

Example:

# Constructing a matrix with duplicate indices

row  = np.array([0, 0, 1, 3])

col  = np.array([0, 2, 1, 3])

data = np.array([3, 1, 2, 1])

coo = coo_matrix((data, (row, col)), shape=(4, 4))


Now that we have our coo_matrix, we store it as a pkl in cooc.pkl.
"""

DATA_PATH = "../data/"

def get_lists(dictionary):
    row = []
    col = []

    for elem in dictionary.keys():
        row.append(elem[0])
        col.append(elem[1])

    return row,col

def main():
    with open(DATA_PATH + 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    counter = 1
    dic = Counter()

    for fn in [DATA_PATH + 'train_pos_full.txt', DATA_PATH + 'train_neg_full.txt']:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        dic[(t,t2)] += 1

                if counter % 10000 == 0:
                    print(counter)
                counter += 1

    row, col = get_lists(dic)
    cooc = coo_matrix((list(dic.values()), (row, col)))

    with open(DATA_PATH + 'cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
        print("Coo_matrix created and stored under 'cooc.pkl' in the data folder")


if __name__ == '__main__':
    main()
