# -*- encoding: utf8 -*-
import re
from sklearn.metrics import accuracy_score, confusion_matrix
from operator import itemgetter

import datetime
import numpy as np
import pandas as pd
import time
from pyvi.pyvi import ViTokenizer
import numpy as np
import os
from sklearn import svm
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import operator

from sklearn.svm import SVC

top = 50
def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"

def clean_str_vn(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[~`@#$%^&*-+]", " ", string)
    def sharp(str):
        b = re.sub('\s[A-Za-z]\s\.', ' .', ' '+str)
        while (b.find('. . ')>=0): b = re.sub(r'\.\s\.\s', '. ', b)
        b = re.sub(r'\s\.\s', ' # ', b)
        return b
    string = sharp(string)
    string = re.sub(r" : ", ":", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def review_to_words(review):
    """
    Function to convert a raw review to a string of words
    :param review
    :return: meaningful_words
    """
    # 1. Convert to lower case, split into individual words
    words = review.lower().split()
    # 2. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    with open('datavn/vietnamese-stopwords-dash.txt', "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    # 3. Remove stop words
    meaningful_words = [w for w in words if not w in array]

    # 4. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)

def print_words_frequency(train_data_features):
    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print("Words in vocabulary:", vocab)

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    print("Words frequency...")
    for tag, count in zip(vocab, dist):
        print(count, tag)
def load_keywords(filename):

    dict = {'ABBR':0,"DESC":1,"ENTY":2,"HUM":3,"LOC":4,"NUM":5}
    array = [[],[],[],[],[],[]]

    with open(filename,'r') as f:
        for line in f:
            line = line.rstrip('\n')

            label, key, quantity = line.split(" ", 3)
            try:
                array[dict[label]].append([key,quantity])
            except:
                print(label)

    return array

def load_data(filename):
    res = []
    col1 = []; col2 = []; col3 = []; col4 = []
    keywords = load_keywords("keyword.txt")
    dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11,'12':12,'14':14,'15':15,'16':16,'17':17,'18':18,'19':19,'20':20,'21':21}
    with open(filename, 'r') as f:
        for line in f:
            label1, p , label2, question = line.split(" ", 3)
            # question = review_to_words(question)
            col1.append(label1)
            col2.append(label2)
            array = question.split()
            try:
                print(range(keywords[dict[label1]]))
            except:
                print(dict[label1],keywords[dict[label1]])
            for index in range(len(keywords[dict[label1]])):

                if keywords[dict[label1]][index][0] in question :

                    for count in range((int)(keywords[dict[label1]][index][1])):
                        array.append(keywords[dict[label1]][index][0])
            question = ' '.join(array)
            col3.append(question)
            if label1 == "ABBR" and filename != "datavn/test":
                for count in range(0,7):
                    col1.append(label1)
                    col2.append(label2)
                    col3.append(question)
        d = {"label1":col1, "label2":col2, "question": col3}
        # d = dict(zip(col1,col3))
        train = pd.DataFrame(d)
        # print train

    return train
train = load_data('datavn/train')
print(train)
def get_tfidf_scores(vectorizer, tfidf_result):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        print('%s : %f' % (item[0], item[1]))

def is_exist(word,sentence):
    array_word = sentence.split()
    return word in array_word

#abbr, desc, entity, hum, loc, num
def count_word_in_document(word,train):
    X_text = train["question"].values
    y_text = train["label1"].values
    array = [0,0,0,0,0,0]
    label = ['ABBR','DESC','ENTY','HUM','LOC','NUM']
    for index in range(len(X_text)):

        if is_exist(word,X_text[index]) :
            for i in range(len(label)) :
                if label[i] == y_text[index]:
                    array[i] = array[i] + 1
                    #print(X_text[index])
    return array
def all_word(train):
    X_text = train["question"].values
    num = set()
    for sentence in X_text:
        k = sentence.split()
        try:
            for index in range(len(k)):
                num.add(k[index])
        except:
            print(num,sentence)
    return num

def dict_count_in_label(train):
    all = all_word(train)
    dict = {}
    for word in all:
        dict[word] = count_word_in_document(word,train)

    return dict
# print(dict_count_in_label(train))

def check_num(array,number):
    if array[number] < 2:
        return False
    else :
        sum = 0
        for i in range(len(array)) :
            if i != number :
                sum = sum +array[i]
            if sum != 0 and array[number]/(sum+array[number]) < 0.7:
                return False
    return True
# dic = {'vĩnh_viễn': [0, 0, 0, 1, 1, 1], 'Thời_gian': [0, 0, 0, 1, 0, 7]}
#
# for num in range(0, 6):
#     print(num)
#     if check_num(dic['vĩnh_viễn'], num):
#         print("ok")
# exit()
def find_key_word(train):
    label = ['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM']
    dict = dict_count_in_label(train)
    key = {0:{},1:{},2:{},3:{},4:{},5:{}}
    for word in dict:
        for num in range(0,6):
            if check_num(dict[word],num) == True :
                key[num][word] = dict[word]
    return key
def sort_dict_idx(dd,idx=2):
    kvs = []
    for key, value in sorted(dd.items(), key=lambda kv: (kv[1][idx], kv[0])):
        kvs.append([key, value])
    return kvs[::-1]
def file_write():

    train = load_data('datavn/train')
    k= find_key_word(train)
    label = ['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM']
    file_name = "keyword.txt"
    fo = open(file_name, 'w')
    for index in range(0,6):
        array = sort_dict_idx(k[index],index)
        for t in range(len(array)) :
            fo.write("%s %s %d\n" % (label[index],array[t][0],array[t][1][index]))






if __name__ == "__main__":

    # load_data("datavn/test")

    # vectorizer = CountVectorizer(analyzer="word",
    #                          tokenizer=None,
    #                          preprocessor=None,
    #                          stop_words=None,
    #                          max_features=1000)
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)


    test = load_data('datavn/test')

    train = load_data('datavn/train')



    print("Data dimensions:", train.shape)
    print("List features:", train.columns.values)
    print("First review:", train["label1"][0], "|", train["question"][0])

    print("Data dimensions:", test.shape)
    print("List features:", test.columns.values)
    print("First review:", test["label1"][0], "|", test["question"][0])
    # train, test = train_test_split(train, test_size=0.2)

    train_text = train["question"].values
    test_text = test["question"].values
    print(train_text,type(train_text),train_text.shape)
    print(train["label1"].values)


    # X_train = vectorizer.fit_transform(train_text)
    vectorizer.fit(train_text)

    X_train = vectorizer.transform(train_text)
    vocal = vectorizer.vocabulary_
    vector = vectorizer.idf_
    sorted_x = sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
    dict ={}
    # print()
    for index in range(len(vector)) :
        dict[sorted_x[index][0]] = vector[sorted_x[index][1]]
    sort_dict = sorted(dict.items(), key=operator.itemgetter(1))
    print(sort_dict)

    X_train = X_train.toarray()
    y_train = train["label1"]
    y_train2 = train["label2"]


    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label1"]
    y_test2 = test["label2"]

    """
    Training
    """

    print("---------------------------")
    print("Training")
    print("---------------------------")
    print(X_train[0])
    #get_tfidf_scores(vectorizer,X_train)

    names = ["RBF SVC"]
    t0 = time.time()
    # iterate over classifiers
    results = {}
    kq = {}
    clf = SVC(kernel='rbf', C=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print y_pred

    print(" accuracy: %0.3f" % accuracy_score(y_test,y_pred))

    print("confuse matrix: \n", confusion_matrix(y_test, y_pred,labels=['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM']))
    # print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))
    #get_tfidf_scores(vectorizer,X_train)

    # print "-----------------------"
    # print "fine grained category"
    # print "-----------------------"
    # clf = SVC(kernel='rbf', C=1000)
    # clf.fit(X_train, y_train2)
    # y_pred = clf.predict(X_test)
    # print y_pred
    #
    # print " accuracy: %0.3f" % accuracy_score(y_test2,y_pred)
    #
