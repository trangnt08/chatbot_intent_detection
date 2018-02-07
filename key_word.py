# -*- encoding: utf8 -*-
import re
import requests
import unicodedata
from tokenizer.tokenizer import Tokenizer
from pyvi.pyvi import ViTokenizer
import pandas as pd
import numpy as np


# tokenizer = Tokenizer()
# tokenizer.run()
def review_to_words(review, filename):
    words = review.lower().split()
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    meaningful_words = [w for w in words if not w in array]
    return " ".join(meaningful_words)

def ngrams(input, n):
  input = input.split(' ')
  output = []
  for i in range(len(input)-n+1):
    output.append(input[i:i+n])
  return output # output dang ['a b','b c','c d']

def ngrams2(input, n):
  input = input.split(' ')
  output = {}
  for i in range(len(input)-n+1):
    g = ' '.join(input[i:i+n])
    output.setdefault(g, 0)
    output[g] += 1
  return output # output la tu dien cac n-gram va tan suat cua no {'a b': 1, 'b a': 1, 'a a': 3}

def ngrams_array(arr,n):
    output = {}
    for x in arr:
        d = ngrams2(x, n)  # moi d la 1 tu dien
        for x in d:
            count = d.get(x)
            output.setdefault(x, 0)
            output[x] += count
    return output

def buid_dict(filename,arr,n,m):
    with open(filename, 'r') as f:
        ngram = ngrams_array(arr, n)
        for x in ngram:
            p = ngram.get(x)
            if p < m:
                f.write(x)

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
    result = result.split(u'\n')[1]
    return result

def is_exist(word,sentence):
    array_word = sentence.split()
    return word in array_word

def count_word_in_document(word,train):
    X_text = train["question"].values
    y_text = train["label"].values
    array = []
    label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15', '6', '17', '18', '19',
             '20', '21']
    for index in range(len(X_text)):

        if is_exist(word,X_text[index]) :
            for i in range(len(label)) :
                if label[i] == y_text[index]:
                    array[i] = array[i] + 1
                    print(X_text[index])
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
def find_key_word(train):
    dict = dict_count_in_label(train)
    key = {0:{},1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{},10:{},11:{},12:{},14:{},15:{},16:{},17:{},18:{},19:{},20:{},21:{}}
    for word in dict:
        for num in range(0,21):
            if check_num(dict[word],num) == True :
                key[num][word] = dict[word]
    return key
def sort_dict_idx(dd,idx=2):
    kvs = []
    for key, value in sorted(dd.items(), key=lambda kv: (kv[1][idx], kv[0])):
        kvs.append([key, value])
    return kvs[::-1]


# def load_keywords(filename):
#
#     dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11,'12':12,'14':14,'15':15,'16':16,'17':17,'18':18,'19':19,'20':20,'21':21}
#     array = []
#
#     with open(filename,'r') as f:
#         for line in f:
#             line = line.rstrip('\n')
#
#             label, key, quantity = line.split(" ", 3)
#             try:
#                 array[dict[label]].append([key,quantity])
#             except:
#                 print(label)
#
#     return array
def load_data(filename, dict):
    col1 = []; col2 = []; col3 = []; col4 = []
    # dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11,
    #         '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21}
    with open(filename, 'r') as f, open(dict, 'w') as f2:
        for line in f:
            if line !='\n':
                label_num, question = line.split("\t")
                # question = clean_str_vn(question)
                label, number = label_num.split("||#")
                question = accent(question)
                # print question
                # question = tokenizer.predict(question)
                question = ViTokenizer.tokenize(question).encode('utf8')
                question = regex_email(question)
                question = regex_phone_number(question)
                question = regex_link(question)
                # print question
                # array = question.split()

                col1.append(label)
                col2.append(number)
                col3.append(question)

        ngram = ngrams_array(col3,2)
        dict_arr = []
        for x in ngram:
            p = ngram.get(x)
            # Neu xuat hien < 1 lan thi ghi vao file f2 de sau nay co the bo di nhung tu it xuat hien
            if p<1:
                dict_arr.append(x)
                f2.write(x+"\n")
        col4 = []
        for q in col3:
            q = review_to_words(q,dict)  # q la 1 cau
            col4.append(q)
        d = {"label":col1, "question": col4}
        train = pd.DataFrame(d)
    return train

def file_write():

    train = load_data('data/ques2.txt','dict/dict1')
    k= find_key_word(train)
    label = ['0','1','2','3','4','5','6','7','8','9','10','11','12','14','15','6','17','18','19','20','21']
    file_name = "data/keyword.txt"
    fo = open(file_name, 'w')
    for index in range(0,21):
        array = sort_dict_idx(k[index],index)
        for t in range(len(array)) :
            fo.write("%s %s %d\n" % (label[index],array[t][0],array[t][1][index]))

if __name__ == '__main__':
    # train = load_data('data/test.txt', 'dict/dict1')
    file_write()