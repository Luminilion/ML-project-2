#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle

"""
Description

FYI:
train_pos.txt is the 100 000 first lines of train_pos_full.txt (which contrains 1 250 000 lines).
Same for train_neg.txt.

###First Part: 
Open the previously constructed vocab.pkl and retrieve the dictionary as vocab.

###Second Part: 
Will do the following for both train_pos.txt and train_neg.txt:

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

    So now, for every index t in tokens, we will go over all the index t2 in tokens and say that
    t (row.append(t)) and t2 (col.append(t2)) co-occur once (data.append(1))

When all of this is done, we will get three lists:
- data : filled with 1's --> the entries of the matrix
- row : filled with ids --> the row indices of the matrix entries
- col : filled with ids --> the column indicies of the matrix entries

Then we will feed these three lists to the coo_matrix function (from scipy.sparse).
coo_matrix[i[k], j[k]] = data[k].

Example:

# Constructing a matrix with duplicate indices

row  = np.array([0, 0, 1, 3, 1, 0, 0])

col  = np.array([0, 2, 1, 3, 1, 0, 0])

data = np.array([1, 1, 1, 1, 1, 1, 1])

coo = coo_matrix((data, (row, col)), shape=(4, 4))

# Duplicate indices are maintained until implicitly or explicitly summed

np.max(coo.data)
1

coo.toarray()
array([[3, 0, 1, 0],
       [0, 2, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 1]])

So here, we see that 0 occurs with 2 once in the array (and we see it as well in row and col)

However, as said in the exemple, duplicate indices are maintained until implicitly summed.

This is why we add : cooc.sum_duplicates at the end.

Now that we have our coo_matrix, we store it as a pkl in cooc.pkl.
"""

DATA_PATH = "../data/"

def main():
    with open(DATA_PATH + 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    counter = 1
    for fn in [DATA_PATH + 'pos_train.txt', DATA_PATH + 'neg_train.txt']:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    with open(DATA_PATH + 'cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
