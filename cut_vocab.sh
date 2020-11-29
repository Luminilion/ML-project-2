#!/bin/bash

# Note that this script uses GNU-style sed as gsed. On Mac OS, you are required to first https://brew.sh/
#    brew install gnu-sed
# on linux, use sed instead of gsed in the command below:

"""
A bit of explanation on what the command line does:

### SED

Will substitute (because of s/) words that start a line (^) 
with one (or many \+) empty space or new line (\s) by nothing (//)
and this for any occurences in a line (/g)

sort -rn: reverse numeric-sort (compare according to string numerical value)

### GREP
Do not take words with a number of occurences of 1,2,3 or 4 followed by a space. 

### cut -d' ' -f2
cut -d allows to cut using a delimiter we specify (here ' ')
-f2 means that you take the second field of the cut (right side the word)
==> getting rid of the word count

Conclusion: 
We end up with all the tweet words (one per line) that appear at least 5 times in the dataset.
"""


cat vocab_full.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > vocab_cut.txt
