'''
File name: glove_template.py
Author: Fatih Mutlu, Loïc Busson, Nicolas Thierry d’Argenlieu
Date created: 23/11/2020
Date last modified: 06/12/2020
Python Version: 3.8.5

glove_template.py is a python file that takes a previously computed co-occurence matrix
and returns the matrix xs containing the word vectors computed with GloVe. 

GloVe uses the following loss function: J = \sum_{i,j}^{V} f(M_{ij})(w_i^Tw_j + b_i + c_j - log(M_{ij}))^2
For now, we took b_i = c_j = 0 --> room to improvement

After considering the gradient respecting...
to continue after the computations
'''

#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random

DATA_PATH = "../data/"

def main():
    print("loading cooccurrence matrix")
    with open(DATA_PATH + 'cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
			logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x

    np.save(DATA_PATH + 'embeddings', xs)


if __name__ == '__main__':
    main()
