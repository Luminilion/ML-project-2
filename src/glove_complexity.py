import pickle
import matplotlib.pyplot as plt
import numpy as np

"""
The goal of this file is to plot the histogram of the number of co-occurences of words to:
1) verify that it is indeed distributed as a power law
2) determine the coefficient beta (slope on log-log scale) to give an estimate of GloVe computational complexity
"""

DATA_PATH = "../data/"

def middle_bins(bins):
    """
    Returns a list with the middle of each bins
    """

    new_bins = []
    for i in range(len(bins)-1):
        new_bins.append(np.exp((np.log(bins[i]) + np.log(bins[i+1]))/2))

    return new_bins

def cleaning(n,bins):
    """
    Take out the 0's from the points to take into account to compute the fitting line
    """
    new_n = []
    new_bins = []

    for i in range(len(n)):
        if n[i] != 0:
            new_n.append(n[i])
            new_bins.append(bins[i])
    
    return new_n, new_bins


def least_squares(y, tx):
    """
    Calculate the least squares solution.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w


with open(DATA_PATH + 'cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)


data = cooc.data

#np.max(data) = 2599902
n, bins, _ = plt.hist(data, bins = np.logspace(0,7,400))

#Center the bins
bins = middle_bins(bins)

#Get rid of the bins with value data = 0
n, bins = cleaning(n, bins)

#list to numpy array
n = np.array(n)
bins = np.array(bins)

#Fitting line
begin = 0
end = len(n)
y = np.log(n[begin:end])
x = np.log(bins[begin:end])
poly = np.ones((len(x), 1))
tx = np.c_[poly, x]

w = least_squares(y,tx)


#Plotting the fitting line
plt.plot(np.exp(x), np.exp(w[0] + w[1]*x))

plt.title('Histogram of the number of co-occurences of words (log-log scale)')
plt.xlabel('Number of co-occurences of words')
plt.ylabel('Frequency')
plt.text(10**3,10**4,"Slope: {:.2f}".format(w[1]))
plt.xscale('log')
plt.yscale('log')
plt.savefig(DATA_PATH + 'hist_glove_complexity.png')

