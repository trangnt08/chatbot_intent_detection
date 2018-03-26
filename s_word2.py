# -*- encoding: utf8 -*-
import re
import requests
import unicodedata
from tokenizer.tokenizer import Tokenizer
from sklearn.externals import joblib
import datetime
import pandas as pd
import time
import os
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from pyvi.pyvi import ViTokenizer
from sklearn.metrics import confusion_matrix


tokenizer = Tokenizer()
tokenizer.run()

def load_model(model):
    print('loading model ...',model)
    if os.path.isfile(model):
        return joblib.load(model)
    else:
        return None

def list_words(mes):
    words = mes.lower().split()
    return " ".join(words)

def regex_email(str):
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', str)
    email = "emails"
    for x in emails:
        str = str.replace(x, email)
    return str

def regex_phone_number(str):
    reg = re.findall("\d{2,4}\D{0,3}\d{3}\D{0,3}\d{3,4}",str)
    # print a
    for x in reg:
        str = str.replace(x,"phone_number")
    return str

def regex_link(str):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str)
    for x in urls:
        str = str.replace(x,"url")
    return str

def clean_doc(question):

    question = regex_email(question)
    question = regex_phone_number(question)
    question = regex_link(question)

    question = unicode(question, encoding='utf-8')
    question = accent(question)
    question = tokenizer.predict(question)  # tu them dau . vao cuoi cau

    rm_junk_mark = re.compile(ur'[?,\.\n]')
    normalize_special_mark = re.compile(ur'(?P<special_mark>[\.,\(\)\[\]\{\};!?:“”\"\'/])')
    question = normalize_special_mark.sub(u' \g<special_mark> ', question)
    question = rm_junk_mark.sub(u'', question)
    question = re.sub(' +', ' ', question)  # remove multiple spaces in a string
    return question

def accent(req):
    data = {'data': req}
    r = requests.post('http://topica.ai:9339/accent', data=data)
    result = r.content
    try:
        result = unicode(result)
    except:
        result = unicode(result, encoding='utf-8')
    # print result
    # result = result.split(u'\n')[1]
    return result


def read_top_term(file):
    with open(file, 'r') as f:
        a = f.read()
        top_term = a.splitlines()
    print top_term
    return top_term


def add_term(str):
    top_term = read_top_term('top_term/x50.txt')
    str = str.lower().split()
    words = [w for w in str]
    a = []
    for i in words:
        if i in top_term:
            for j in range(0,5):
                a.append(i)
    ques = words + a
    return " ".join(ques)

def build_data(list_ques):
    list_q = []
    for q in list_ques:
        q = add_term(q)
        list_q.append(q)
    return list_q

def load_data(filename):
    yn_label = []
    col1 = []; col2 = []; col3 = []
    with open(filename, 'r') as f:
        for line in f:
            if line !='\n':
                label_num, question = line.split("\t")
                label, number = label_num.split("||#")
                question = clean_doc(question)
                col1.append(label)
                col2.append(number)
                col3.append(question)
        if filename == 'data/sum_ques.txt':
            with open('data/question2.txt','r') as f3:
                for line in f3:
                    if line != '\n':
                        label_num, question = line.split("\t")
                        label, number = label_num.split("||#")
                        l1,l2 = label.split("-")
                        question = clean_doc(question)
                        yn_label.append(l1)
                        col1.append(l2)
                        col2.append(number)
                        col3.append(question)

            # col4.append(q)
        d = {"label":col1, "question": col3}
        train = pd.DataFrame(d)
        if filename == 'data/sum_ques.txt':
            joblib.dump(train,'model2/train1.pkl')
        else:
            joblib.dump(train,'model2/test1.pkl')
    return train

def training():
    train = load_model('model2/train1.pkl')
    if train == None:
        train = load_data('data/sum_ques.txt')
    print "---------------------------"
    print "Training"
    print "---------------------------"
    vectorizer = load_model('model2/vectorizer1.pkl')
    if vectorizer is None:
       vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    train_text = train["question"].values
    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    X_train = X_train.toarray()
    y_train = train["label"]
    joblib.dump(vectorizer, 'model2/vectorizer1.pkl')
    fit1(X_train, y_train)

def fit1(X_train,y_train):
    uni_big = SVC(kernel='rbf', C=1000)
    uni_big.fit(X_train, y_train)
    joblib.dump(uni_big, 'model2/uni_big1.pkl')
def test_file():
    uni_big = load_model('model2/uni_big1.pkl')
    if uni_big is None:
        training()
    uni_big = load_model('model2/uni_big1.pkl')
    vectorizer = load_model('model2/vectorizer1.pkl')
    test = load_model('model2/test1.pkl')
    if test is None:
        test = load_data('data/test2.txt')
    test_text = test["question"].values
    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label"]
    y_pred = uni_big.predict(X_test)
    print y_pred

    print " accuracy: %0.3f" % f1_score(y_test, y_pred, average='weighted')
    print "confuse matrix: \n", confusion_matrix(y_test, y_pred,
                                                 labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
                                                         "12", "14", "15", "16", "17", "18", "19", "20", "21","22","23","24","25","26","27"])

if __name__ == '__main__':
    test_file()
