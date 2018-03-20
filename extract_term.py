# -*- encoding: utf8 -*-
import re
import requests
import unicodedata
from tokenizer.tokenizer import Tokenizer
from pyvi.pyvi import ViTokenizer
from sklearn.externals import joblib
import datetime
import pandas as pd
import time
import os
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from pyvi.pyvi import ViTokenizer
from sklearn.metrics import confusion_matrix
import operator

tokenizer = Tokenizer()
tokenizer.run()

def load_model(model):
    print('loading model ...',model)
    if os.path.isfile(model):
        return joblib.load(model)
    else:
        return None



# def regex_dau(str):
#     str = str.replace(". ", " ")
#     str = str.replace("? ", " ")
#     str = str.replace(".\n", "\n")
#     str = str.replace("?\n", "\n")
#     str = str.replace(".", "")
#     str = str.replace("?", "")
#     str = str.replace("\n", " \n")
#     return str

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

def inverse_document_frequencies(tokenized_documents):
    # tokenized_documents: list cac list cua nhieu document  [[w1...wn]...[w1...wn]]
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    # sublist: list cua 1 document
    # item: 1 tu trong sublist
    # all_tokens_set: tap cac tu trong tat ca document, cac tu ko lap lai. vd: set(['a','b','c','d'])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        # tra ve true neu tkn thuoc doc, thuc hien ham lambda tren tat ca document
        print contains_token
        # contains_token: list
        idf_values[tkn] = 1 + math.log(len(tokenized_documents) / (sum(contains_token)))
    print idf_values
    return idf_values

def load_documents(filename):
    yn_label = []
    col1 = [];
    col2 = [];
    col3 = []
    with open(filename, 'r') as f:
        for line in f:
            if line != '\n':
                label_num, question = line.split("\t")
                label, number = label_num.split("||#")

                question = regex_email(question)
                question = regex_phone_number(question)
                question = regex_link(question)
                question = tokenizer.predict(question)

                col1.append(label)
                col2.append(number)
                col3.append(question)
    d = {"label":col1, "question":col3}
    return d


def document_frequencies(documents,n,file):
    df_values = {}
    tokenize = lambda doc: doc.lower().split(" ")
    tokenized_documents = [tokenize(d) for d in documents]
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    # print all_tokens_set
    # sublist: list cua 1 document
    # item: 1 tu trong sublist
    # all_tokens_set: tap cac tu trong tat ca document, cac tu ko lap lai. vd: set(['a','b','c','d'])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc and tkn!="" and tkn!=" ", tokenized_documents)
        # tra ve true neu tkn thuoc doc, thuc hien ham lambda tren tat ca document
        df_values[tkn] = sum(contains_token)
    # sorted_x = sorted(df_values.items(), key=operator.itemgetter(1))
    sorted_x = sorted(df_values.items(), key=lambda x:x[1],reverse=True)
    top_term = sorted_x[0:n]
    with open(file,"w") as f:
        for i in top_term:
            f.write(i[0].encode('utf-8')+" : "+str(i[1])+"\n")
            # print top_term
    return top_term

# loai bo cac tu giong trong class khac
def get_list_term(documents):
    df_values = {}
    tokenize = lambda doc: doc.lower().split(" ")
    tokenized_documents = [tokenize(d) for d in documents]
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    # print all_tokens_set
    # sublist: list cua 1 document
    # item: 1 tu trong sublist
    # all_tokens_set: tap cac tu trong tat ca document, cac tu ko lap lai. vd: set(['a','b','c','d'])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc and tkn!="" and tkn!=" ", tokenized_documents)
        # tra ve true neu tkn thuoc doc, thuc hien ham lambda tren tat ca document
        df_values[tkn] = sum(contains_token)
    # sorted_x = sorted(df_values.items(), key=operator.itemgetter(1))
    sorted_x = sorted(df_values.items(), key=lambda x:x[1],reverse=True)
    top_term = sorted_x[0:50]
    return top_term

def add_term(str,top_term):
    list_term=[]
    for i in top_term:
        list_term.append(i[0])
    # print list_term
    str = str.lower().split()
    words = [w for w in str]
    a = []
    for i in words:
        if i in list_term:
            for j in range(0,5):
                a.append(i)
    ques = words + a
    return " ".join(ques)

def top_term_all_class(list_ques,n,file):
    top_term = get_list_term(list_ques)

def build_data(list_ques,n,file):
    top_term = document_frequencies(list_ques,n,file)
    list_q = []
    for q in list_ques:
        q = add_term(q,top_term)
        list_q.append(q)
    return list_q



def get_documents(filename):
    yn_label = []
    col1 = []
    col2 = []
    col3 = []
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15', '16', '17', '18', '19',
              '20', '21', '22', '23', '24', '25', '26', '27']
    list_l = []
    list_n = []
    list_q = []
    len_l = len(labels)
    for i in range(len_l):
        list_l.append([])
        list_n.append([])
        list_q.append([])
    list_l.append([])
    list_n.append([])
    list_q.append([])
    with open(filename, 'r') as f:
        for line in f:
            if line != '\n':
                label_num, question = line.split("\t")
                label, number = label_num.split("||#")
                question = regex_email(question)
                question = regex_phone_number(question)
                question = regex_link(question)
                # question = regex_dau(question)

                question = unicode(question,encoding='utf-8')
                question = tokenizer.predict(question)  # tu them dau . vao cuoi cau

                rm_junk_mark = re.compile(ur'[?,\.\n]')
                normalize_special_mark = re.compile(ur'(?P<special_mark>[\.,\(\)\[\]\{\};!?:“”\"\'/])')
                question = normalize_special_mark.sub(u' \g<special_mark> ', question)
                question = rm_junk_mark.sub(u'', question)
                question = re.sub(' +',' ',question)    #remove multiple spaces in a string

                for l in labels:
                    if l == label:
                        i = int(l)
                        list_l[i].append(l)
                        list_n[i].append(number)
                        list_q[i].append(question)
        for i in range(len_l+1):
            col1 = col1 + list_l[i]
            col2 = col2 + list_n[i]

        d_col = {'0':list_q[0], '1':list_q[1], '2':list_q[2], '3':list_q[3], '4':list_q[4],
        '5':list_q[5], '6':list_q[6], '7':list_q[7], '8':list_q[8], '9':list_q[9], '10':list_q[10],
        '11':list_q[11], '12':list_q[12], '14':list_q[14], '15':list_q[15], '16':list_q[16], '17':list_q[17],
        '18':list_q[18], '19':list_q[19], '20':list_q[20], '21':list_q[21], '22':list_q[22],
        '23':list_q[23], '24':list_q[24], '25':list_q[25], '26':list_q[26], '27':list_q[27]}
        d = {"label": col1, "question": d_col}
    return d

def load_data1(filename,n):
    d = get_documents(filename)
    d_doc = d.get("question")

    list_docs = d_doc.values()
    list_label = d_doc.keys()
    all_docs = []
    all_label = []
    # print list_docs
    # docs: cac ques cung 1 lop; list_docs: tat ca cac ques o tat ca cac lop
    for i,j in zip(list_docs,list_label):
        k = "top_term/" + str(j) + '.txt'
        n_docs = build_data(i,n,k)
        # print i
        for i in n_docs:
            all_docs.append(i)
            all_label.append(j)

    print all_label

def load_data2(filename,n):
    d = get_documents(filename)
    # print "d", d
    d_doc = d.get("question")

    list_docs = d_doc.values()
    # print list_docs
    list_label = d_doc.keys()
    # docs: cac ques cung 1 lop; list_docs: tat ca cac ques o tat ca cac lop
    all_term = []

    for i,j in zip(list_docs,list_label):
        list_term = get_list_term(i)
        all_term.append(list_term)

    l1 = []
    for l_term in all_term:
        l2 = []
        for i in l_term:
            l2.append(i[0])
        l1.append(l2)

    seta=set([])
    for i in l1:
        seta = seta.union(set(i))   # lay tat ca cac tu

    l3 = []
    print all_term
    for i in range(len(l1)):
        setx = set([])
        for j in range(len(l1)):
            # print i,j
            if i!=j:
                setx =setx.union(l1[j]) # thuoc cac lop con lai tru lop i
        x=list(setx)

        l4 = []
        for e in all_term:

            for f in e:
                # print f
                if not f[0] in x:
                    l4.append(f)

        l3.append(l4)
        setx = setx.intersection([])

    with open('top_term/30.txt', "w") as f:
        for i, k in zip(l3, list_label):
            f.write(str(k) + "\n")
            z = 0
            for j in i:
                if j[0] != "" and z<20:
                    f.write(j[0].encode('utf-8') + " : " + str(j[1]) + "\n")
                    z += 1
            f.write("\n")


if __name__ == '__main__':
    load_data1('data/sum_ques.txt',20)
    # load_data2('data/sum_ques.txt',10)