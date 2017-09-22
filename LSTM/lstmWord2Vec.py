"""
The architecture implemented in this file consists of training 2 separate
LSTM networks using the embeddings from Google's 300B corpus. The output of
these 2 networks is then fed to a similarity estimator to predict the
method creates 2 LSTM

"""


# from keras import backend as  K
# import keras as keras
# from keras.layers.merge import concatenate
# import numpy as np
import string
import sys
import pandas as pd
from gensim.models import KeyedVectors
from nltk.corpus import stopwords


cmdLineArgs = sys.argv[1:]
dataset = cmdLineArgs[0]
word2vec = cmdLineArgs[1]

# Extract data from the source file
dataframe = pd.read_csv(dataset, usecols=['question1','question2','is_duplicate'])
new_dataframe = dataframe.dropna()
question1 =  new_dataframe['question1']
question2 = new_dataframe['question2']
label = new_dataframe['is_duplicate']

print("\nTraining Samples: {}".format(len(label)))                #Training Samples:       404288
print("Similar Queries: {}".format(label.value_counts()[1]))      #Similar Query pairs:    149263
print("Dis-similar Queries: {}".format(label.value_counts()[0]))  #Dissimilar Query pairs: 255025

# Prepare word embedding matrix using Google's Word2Vec
GVectors = KeyedVectors.load_word2vec_format(word2vec,binary='True')
print("\nSimilarity test for words \"2000\" and \"2001\" {}".format(GVectors.wv.similarity('lose','loose')))

# Lets do some data processing
def cleanData(input):

    #Lowercase
    new_input = input.lower()

    #Remove punctuations
    #predicate = lambda x: x not in string.punctuation
    #unpunkt = filter(predicate,new_input)

    #Remove stop words(Lets make this optional)
    splits = new_input.split()
    predicate = lambda x: x not in stopwords.words('english')
    filtered = filter(predicate, splits)

    return ' '.join(filtered)

# Let's see which words from the corpus is not available in Vector form
def findWordsNonVectorWords(word2vec,data):
    for sentence in data:
        sents = sentence.lower().split()
        OOV = [word for word in sents if word not in word2vec.vocab]
        print(OOV)


findWordsNonVectorWords(GVectors,question1)
findWordsNonVectorWords(GVectors,question2)
