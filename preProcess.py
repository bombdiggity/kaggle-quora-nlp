"""
Process the Quora Dataset.


"""

import sys
import os
import pandas as pd

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

    createQuestionFile("question1.txt",qid1,q1)
    createQuestionFile("question2.txt",qid2,q2)

    createLabelFile("label.txt",dup)

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
        fq.write(str(question))
        fq.write("\n")

    fq.close()

if __name__ == "__main__":
    processDataset(sys.argv[1:])