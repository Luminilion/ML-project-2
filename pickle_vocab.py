#!/usr/bin/env python3
import pickle

"""
pickle_vocab.py is used to open vocab_cut.txt 'containing all the tweet tokens
with a least 5 occurences) and store each token in a dictionnary (vocab) where the key
is the token, and the value is an identifier

Example: vocab = {'<user>' : 0, '!' : 1, 'i' : 2 ,...,'classpack': 98483, 'classifying': 98484, 'clasps': 98485, 'clasic': 98486, 'clase': 98487, "clarkson's": 98488, 'clarksies': 98489,...}

In the second part of the main function, it will store that dictionary in a pickle file
called vocab.pkl.

Therefore, this function is used to store vocab_cut.txt as a dictionnary with identifiers in a pkl format.
"""

def main():
    vocab = dict()
    with open('vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
