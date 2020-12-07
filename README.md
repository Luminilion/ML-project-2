# TEAM MLGang - Project 2 - 2020 Fall Machine Learning (CS-433)

[Full assignment guideline pdf here](https://github.com/epfml/ML_course/blob/master/projects/project2/project2_description.pdf)

## Report

Here is our report in pdf format: [TBD]()

## Team

- Nicolas Thierry d'Argenlieu (nicolas.thierrydargenlieu@epfl.ch)
- Loïc Busson (loic.busson@epfl.ch)
- Fatih Mutlu (fatih.mutlu@epfl.ch)

## Requirements

*To do: list all the libraries with version to be used*

## Organisation

There are five folders: `data`, `plot`, `project_description`, `src` and `submission`.

### 1) Data

In the `data` folder, you will find the `twitter-datasets.zip`. You should unzip it locally and pour its content directly in the `data` folder. This being done the `data` folder should contain:

- `sample_submission.csv`: an example of submission given by AIcrowd
- `test_data.txt`: the tweets to classify
- `train_neg_full.txt`: negative tweets to train on
- `train_neg.txt`: small portion of `train_neg_full.txt` (for testing purposes)
- `train_pos_full.txt`: positive tweets to train on
- `train_pos.txt`: small portion of `train_pos_full.txt` (for testing purposes)

We have also put `embeddings.npy` so that one can skip the **Computing the co-occurrence matrix** and **Computing the Word Embeddings** described below.

### 2) Plots

In the `plots` folder, you will find graphs plotted for:

- optimization of hyperparameters
- evaluating predictive accuracy and comparing to BERT's
- ...

### 3) Project_description

In the `project_description` folder, you will find the documents related to the assignment such:

- `Assignment.md`: Project Text Sentiment Classification Assignment
- `project2_description.pdf`: Overall Project 2 description pdf

### 4) Src

In the `src` folder, you will find our code: *more precise description can be found directly in the files Docstring*

- `build_vocab.sh`: creates `vocab_full.txt` in the `data` folder containing all the (unique) words sorted along with their number of occurrences appearing in `train_pos_full.txt` and `train_neg_full.txt`.

- `cut_vocab.sh`: creates `vocab_cut.txt` in the `data` folder with the list of words occurring at least 5 times in `vocab_full.txt`, by the inverse order of the number of occurrences.

- `pickle_vocab.py`: converts `vocab_cut.txt` to a dictionnary with identifiers and stores it as `vocab.pkl` in the `data` folder (pkl format).

- `cooc.py`: uses `vocab.pkl` to create and store the co-occurrence matrix in `cooc.pkl` in the `data` folder.

- `glove_template.py`: uses the co-occurrence matrix stored in `cooc.pkl` to compute word vectors using GloVe and stores the result in `embeddings.npy` in the `data` folder.

- `Project2.ipynb`: notebook used to train and tune the classifier and make predictions about the `test_data.txt` file. It will output `submission.csv` that will be then stored in the submission `folder`.

### Submission

In the `submission` folder, you will find our latest submission:

- `submission.csv`: latest submission (classification prediction of `test_data.txt`)


## Recreate final solution

First, unzip the .zip file in the `data` folder and pour its content directly in the `data` folder.

**Computing the co-occurrence matrix**

Run in the terminal (while CWD is inside `/src/`): *this can take a few minutes*

```
./build_vocab.sh
./cut_vocab.sh
python pickle_vocab.py
python cooc.py
```

Note that you might have to give permission access to the SH files.
To do this:

```
chmod u+x build_vocab.sh
chmod u+c cut_vocab.sh
```

After completion, you should find in the `data` folder `cooc.pkl` along with new .pkl and .txt files (intermediary results).

**Computing the Word Embeddings**

Run in the terminal (while CWD is inside `/src/`): *this can take a long time*

```
python glove_template.py
```

After completion, you should find in the `data` folder `embeddings.npy` encoding the Word Embeddings computed. 

**Training the model**

You are now ready to train the model. To do this, open the `Project2.ipynb` notebook and follow the instructions in Markdown. 

**Submission**

The model now trained, it is possible to make predictions by following the steps describe in `Project2.ipynb`.

The predictions should be in the `submission` folder saved as `submission.csv`.


