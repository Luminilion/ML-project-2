#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random

"""
Now that we have the co-occurence matrix (not calculated on the full dataset) and the vocabulary,
we want to compute an embedding vector for each word in the vocabulary.

For this, we want to represent context words and words in a lower dimension space (here 20)
in the ndarray xs and ys (randomly initialized at first). The goal is to change them such
that if the word on the i-th row of xs and the context on the j-th row of ys are similar (ie co-occur a lot), then
the product of these two vectors is high (and really low if they differ ie not occur together).
In other words, we are looking for a decomposition of the co-occurance matrix of shape (n,n) into
two matrices xs and ys of shape (n,embedding_dim) such that:
cooc = xs @ ys.T

"""

def main():
    print("\nloading cooccurrence matrix")
    with open('cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries\n".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("\ninitializing embeddings\n")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    print("Visualizing cooc:")
    print(cooc.row)
    print(cooc.col)
    print(cooc.data)
    print(f"Meaning that {cooc.row[1]} and {cooc.col[1]} co-occurs {cooc.data[1]} times\n")

    """
    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):"""

			# fill in your SGD code here, 
			# for the update resulting from co-occurence (i,j)
		

    #np.save('embeddings', xs)


if __name__ == '__main__':
    main()
