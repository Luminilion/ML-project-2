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

To start with, we will take L = sum((Cij - xs@ys.T)**2) + regularization term as a loss function
"""

def sgd_grad(cooc_data, xs, ys, ix, jy, lambda_x, lambda_y):
    """
    Computes the gradient of the loss function respective to xs and to ys

    Parameters
    ----------
    cooc: ndarray (n,n) 
    xs: ndarray (n,d)
    ys: ndarray (n,d)
    ix: int
    jy: int

    Returns
    -------
    grad_xs, grad_ys: ndarray (d,)
    """

    e = xs[ix] @ (ys[jy].T) - cooc_data
    grad_xsi = 2*e*ys[jy] + 2*lambda_x*xs[ix]
    grad_ysj = 2*e*xs[ix] + 2*lambda_y*ys[jy]

    return grad_xsi, grad_ysj

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

    eta = 0.01
    alpha = 10

    epochs = 10

    print("Visualizing cooc:")
    print(cooc.row)
    print(cooc.col)
    print(cooc.data)
    print(f"Meaning that {cooc.row[1]} and {cooc.col[1]} co-occurs {cooc.data[1]} times\n")


    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):

            grad_xsi, grad_ysj = sgd_grad(n, xs, ys, ix, jy, alpha, alpha)
            
            xs[ix] = xs[ix] - eta * grad_xsi
            ys[jy] = ys[jy] - eta * grad_ysj

			# fill in your SGD code here, 
			# for the update resulting from co-occurence (i,j)
		

    #np.save('embeddings', xs)
    print(xs[1]@(ys[1].T))

if __name__ == '__main__':
    main()
