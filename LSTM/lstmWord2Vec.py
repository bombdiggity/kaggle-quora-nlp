"""
The architecture implemented in this file consists of training 2 separate
LSTM networks using the embeddings from Google's 300B corpus. The output of
these 2 networks is then fed to a similarity estimator to predict the
method creates 2 LSTM

"""


# from keras import backend as  K
# import keras as keras
# from keras.layers.merge import concatenate
import numpy as np
import sys
import pandas as pd
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.models import Model
from contractionsExpander import expandContractions
import re


MAX_WORDS = 20000
MAX_LENGTH = 100
EMBED_DIM = 300
TEST_SPLIT = 0.8

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
print("\nSimilarity test for words lose and loose {}".format(GVectors.wv.similarity('lose','loose')))

# Lets do some data processing
def cleanData(input):

    #Remove punctuations
    #predicate = lambda x: x not in string.punctuation
    #unpunkt = filter(predicate,new_input)

    #Just keep alpha numerals and some punctuations
    new_input = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", input.lower())

    #Remove stop words(Lets make this optional)
    splits = new_input.split()
    predicate = lambda x: x not in stopwords.words('english')
    filtered = filter(predicate, splits)

    #Remove Contractions
    ret = expandContractions(' '.join(filtered))

    return ret

cleanQuestion1 = [cleanData(sentence) for sentence in question1]
cleanQuestion2 = [cleanData(sentence) for sentence in question2]

# Tokenize the data
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(cleanQuestion1 + cleanQuestion2)

sequence1 = tokenizer.texts_to_sequences(cleanQuestion1)
sequence2 = tokenizer.texts_to_sequences(cleanQuestion2)

print("Total number of tokenized words: {}".format(tokenizer.word_counts))

pad_sequence1 = pad_sequences(sequences=sequence1, maxlen=MAX_LENGTH)
pad_sequence2 = pad_sequences(sequences=sequence2, maxlen=MAX_LENGTH)

# Embedding Layer
word_index = tokenizer.word_index
word_items = word_index.items()
totalVocab = len(word_index)
embedding_matrix = np.zeros((totalVocab,EMBED_DIM))

for word, index in word_items:
    if word in GVectors.vocab:
        embedding_matrix[index] = GVectors.word_vec(word)

embed_layer = Embedding(input_dim=totalVocab,
                        output_dim=EMBED_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_LENGTH,
                        trainable=False)

# Data splits
shuffle_idx = np.random.permutation(len(pad_sequence1))
train_idx = shuffle_idx[:int(TEST_SPLIT * len(pad_sequence1))]
test_idx = shuffle_idx[-int(1-(TEST_SPLIT*len(pad_sequence1))):]

train1_data = [ pad_sequence1[idx] for idx in train_idx ]
train2_data = [ pad_sequence2[idx] for idx in train_idx ]
train_label = [ label[idx] for idx in train_idx ]

test1_data = [ pad_sequence1[idx] for idx in test_idx ]
test2_data = [ pad_sequence2[idx] for idx in test_idx ]
test_label = [ label[idx] for idx in test_idx ]

# Define LSTM Model Architecture
lstm_layer = LSTM(units=128,dropout=0.2,recurrent_dropout=0.2)

input1 = Input(shape=(MAX_LENGTH), dtype='int32')
embedded_input1 = embed_layer(input1)
lstm1_input = (embedded_input1)

input2 = Input(shape=(MAX_LENGTH), dtype='int32')
embedded_input2 = embed_layer(input2)
lstm2_input = (embedded_input2)

layer = Concatenate([lstm1_input,lstm2_input])
layer = Dropout(rate=0.2)(layer)
layer = BatchNormalization(layer)

output = Dense(1,activation='sigmoid')(layer)

# Model Functional API
model = Model(inputs=[train1_data,train2_data], outputs=[output])
model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])
print("Start Training the model")
history = model.fit([train1_data,train2_data],train_label, batch_size=2048, epochs=100, shuffle=True)

print("Model Training Complete. History: {}".format(history.history))

print("Start Prediction")
pred_result = model.predict([test1_data, test2_data], batch_size=2048,verbose=1)

result_arr = [new_dataframe.ix[idx] for idx in test_idx]
result_df = pd.DataFrame(result_arr)
result_df["predictions"] = pred_result

print("Writing predictions to file")
result_df.to_csv('prediction.csv', index=False, encoding='utf-8')



def getWordVectors(word2vec,input):

    vector = [word2vec.word_vec(word) if word in word2vec.vocab else 0 for word in input ]



#(Optional) Let's see which words from the corpus is not available in Vector form
def findWordsNonVectorWords(word2vec,data):
    for sentence in data:
        sents = sentence.lower().split()
        OOV = [word for word in sents if word not in word2vec.vocab]
        print(OOV)


#findWordsNonVectorWords(GVectors,cleanQuestion1)
#findWordsNonVectorWords(GVectors,question2)