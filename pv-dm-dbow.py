"""
Paragraph Vector: Distributed Memory
Paragraph Vector: Distributed BOW
"""

import sys
import os
import numpy as np
from datetime import datetime, time, date
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression



PV_DM   = 'impr_doc2vec_def_dm'
PV_DBOW = 'impr_doc2vec_def_bow'

def prepData(args):

    q1_file = args[0]
    q2_file = args[1]
 
    q1_words, q1_tags = createTaggedDocuments(q1_file)
    q2_words, q2_tags = createTaggedDocuments(q2_file)
    #docs = createTaggedDocuments(q1_file) + createTaggedDocuments(q2_file)

    #return docs
    return (q1_words, q1_tags, q2_words, q2_tags)

def createTaggedDocuments(fname):

    taggedDocs = []
    tags = []
    fp = open(fname,'r')
    lines = fp.readlines()
    for line in lines:
        groups = line.strip().split(":")
        tags.append(groups[0])
        td = TaggedDocument(words = groups[1].split(), tags = [groups[0]])
        taggedDocs.append(td)

    return (taggedDocs,tags)


def train_PV_DM(docs):

    print("############### Start training PV-DM ##################")
    start = datetime.utcnow()
    model = Doc2Vec(documents = docs, size=100, window=5, min_count=5, iter=25)
    end   = datetime.utcnow()
    duration = datetime.combine(date.min, end.time()) - datetime.combine(date.min, start.time())
    print("Training PV-DM  Completed in: {} seconds".format(str(duration.seconds)))
    model.save(PV_DM)
    return

def train_PV_BOW(docs):

    print("############### Start training PV-BOW ##################")
    start = datetime.utcnow()
    model = Doc2Vec(documents = docs, dm=0, dbow_words=1, size=100, window=5, min_count=5, iter=25)
    end = datetime.utcnow()
    duration = datetime.combine(date.min, end.time()) - datetime.combine(date.min, start.time())
    print("Training PV-BOW  Completed in: {} seconds".format(str(duration.seconds)))
    model.save(PV_DBOW)
    return


def compVectors(q1_tags, q2_tags, modelName):

    print("############## Finding Vector Similarity Scores for: {}".format(modelName))
    fw = open("impr_Sims_q1-q2_"+modelName,'w+')

    model = Doc2Vec.load(modelName)
    for q1, q2 in zip(q1_tags,q2_tags):
        #print(model.docvecs.similarity(q1,q2))
        fw.write(str(model.docvecs.similarity(q1,q2)))
        fw.write("\n")

    fw.close()


def testSameVectors(modelName):

    #doc20 = [u'Why',u'do',u'rockets',u'look',u'white?']
    doc20 = [u'rockets',u'look',u'white']

    print("############## {}: Checking Inferred Vector similarity for test vector".format(modelName))

    model = Doc2Vec.load(modelName)
    inferred_vector = model.infer_vector(doc20)
    print(model.docvecs.most_similar([inferred_vector],topn=10))


def testClassifierLogistic(modelName):

    print("############## Begin training LR Classifier for: {}".format(modelName))

    train_x = np.zeros((12345,100))
    train_y = np.zeros(12345)

    lr_classifier = LogisticRegression()
    lr_classifier.fit(train_x,train_y)

    lr_classifier.score()


def findMean(identifier, input):

    val = reduce((lambda x,y:len(x.words)+len(y.words)),input)
    print ("Mean of {} sentence length is {} words.".format(identifier,(val / len(input))))


if __name__ == '__main__':

    taggedDocs = prepData(sys.argv[1:])
    q1_words, q1_tags, q2_words, q2_tags = prepData(sys.argv[1:])
    taggedDocs = q1_words + q2_words

    #train_PV_DM(taggedDocs)
    #train_PV_BOW(taggedDocs)

    # if not os.path.isfile(PV_DM) and not os.path.isfile(PV_DBOW):
    #
    #     train_PV_DM(taggedDocs)
    #     train_PV_BOW(taggedDocs)

    compVectors(q1_tags, q2_tags, PV_DM)
    compVectors(q1_tags, q2_tags, PV_DBOW)

    #testSameVectors(PV_DM)
    #testSameVectors(PV_DBOW)

    #findMean("Question1", q1_words)
    #findMean("Question2", q2_words)

    #testClassifierLogistic(PV_DM)
    #testClassifierLogistic(PV_DBOW)
