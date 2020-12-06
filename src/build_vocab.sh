#!/bin/bash

# Note that this script uses GNU-style sed as gsed. On Mac OS, you are required to first https://brew.sh/
#    brew install gnu-sed
# on linux, use sed instead of gsed in the command below:

# Outputs all the (unique) words sorted along with their number of occurrences 

"""
Explanations:

### CAT:
cat test1 test2 test3 | sort > test4 will create a file test4 with the contents of test123 sorted.
Here, by alphabetical order.

### SED:
sed 's/': means we are doing a substitution
    's/ /': means we are substituting spaces
    's/ /\n': by \n (newline)

's/ /\n/g': means that we are replacing all occurences of the pattern in a line 
(by default, without the flag /g, it's only the first occurence in the line that is substituted)

Because all tweets have already been pre-processed so that all tokens are separated by a single whitespace, 
what we've done so far is creating a .txt with one token per line.

### GREP:

Grep command is used to search text. It searches the given files for lines containing a match to the given strings or words

The -v option is used to print inverts the match, meaning that it matches only the lines that do not contain the given word

So here, it's taking all lines (so all words) that don't contain '^\s*$'.
'^\s*$' : this is a regex expression saying:
- ^ ==> matches only if the word is at the begining of the string of a line
- \s matches a whitespace character (space, newline,...)
- * matches the preceding element zero or more times
-$ matches the ending position of the string or the position just before a string-ending newline.

Therefore, this command allows to take everything except for lines that have only whitespaces.

### UNIQ -c
uniq will compare lines of a file and display only one line for multiple occurence of identical lines
the -c allows to get count of unique lines. 


CONCLUSION
So in the end, we will have lines of token of the form 'n token' with n being the number of time this token appeared.
"""

cat ../data/train_pos_full.txt ../data/train_neg_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ../data/vocab_full.txt
