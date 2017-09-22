"""
Process the Quora Dataset.

Usage: python preProcessImpr.py train.csv
"""

import sys
import os
import pandas as pd
import string
from nltk.corpus import stopwords

def processDataset(input):

    if os.path.isfile(input[0]) == False:
        print("Input Dataset does not exist")

    """
    Reading CSV using Pandas.
    """
    df = pd.read_csv(input[0],usecols=["qid1","qid2","question1","question2","is_duplicate"])

    new_df = df.dropna()
    qid1 = new_df["qid1"]
    qid2 = new_df["qid2"]
    q1 = new_df["question1"]
    q2 = new_df["question2"]
    dup = new_df["is_duplicate"]

    createQuestionFile("impr_question1.txt",qid1,q1)
    createQuestionFile("impr_question2.txt",qid2,q2)

    createLabelFile("impr_label.txt",dup)

def createLabelFile(fname, data):

    fl = open(fname, "w+")
    for label in data:
        fl.write(str(label))
        fl.write("\n")
    fl.close()

def createQuestionFile(fname,qids,sentences):

    fq = open(fname, "w+")
    for id, question in zip(qids, sentences):
        fq.write(str(id))
        fq.write(":")
        fq.write(str(cleanData(question)))
        fq.write("\n")

    fq.close()


def calcDatasetStats(input):

    df = pd.read_csv(input[0], usecols=["qid1", "qid2", "question1", "question2", "is_duplicate"])
    new_df = df.dropna()
    dup = new_df["is_duplicate"]

    print("-------- Dataset Stats ----------")
    print("Question pairs that are similar: {}".format(dup.value_counts()[1]))
    print("Question pairs that are NOT similar: {}".format(dup.value_counts()[0]))

def cleanData(input):

    #Lowercase
    new_input = input.lower()
    print(new_input)

    #Remove punctuations
    predicate = lambda x: x not in string.punctuation
    unpunkt = filter(predicate,new_input)

    #Remove stop words(Lets make this optional)
    splits = unpunkt.split()
    predicate = lambda x: x not in stopwords.words('english')
    filtered = filter(predicate, splits)

    return ' '.join(filtered)


if __name__ == "__main__":
    processDataset(sys.argv[1:])
    calcDatasetStats(sys.argv[1:])