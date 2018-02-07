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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from pyvi.pyvi import ViTokenizer
from sklearn.metrics import confusion_matrix

tokenizer = Tokenizer()
tokenizer.run()

def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"

def load_model(model):
    print('loading model ...')
    if os.path.isfile(model):
        return joblib.load(model)
    else:
        return None

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
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def review_to_words(review, filename):
    """
    Function to convert a raw review to a string of words
    :param review
    :return: meaningful_words
    """
    # 1. Convert to lower case, split into individual words
    words = review.lower().split()
    # 2. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    # 3. Remove stop words
    meaningful_words = [w for w in words if not w in array]

    # 4. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)

def list_words(mes):
    words = mes.lower().split()
    return " ".join(words)

def review_to_words2(review, filename,n):
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    words = [' '.join(x) for x in ngrams(review, n)]
    meaningful_words = [w for w in words if not w in array]
    return build_sentence(meaningful_words)

def word_clean(array, review):
    words = review.lower().split()
    meaningful_words = [w for w in words if w in array]
    return " ".join(meaningful_words)

def print_words_frequency(train_data_features):
    # Take a look at the words in the vocabulary
    vectorizer = load_model('model/vectorizer.pkl')
    if vectorizer == None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    vocab = vectorizer.get_feature_names()
    print "Words in vocabulary:", vocab

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    print "Words frequency..."
    for tag, count in zip(vocab, dist):
        print count, tag

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

# def build_dict(arr,n,m):
#     d={}
#     ngram = ngrams_array(arr,n)
#     for x in ngram:
#         p = ngram.get(x)
#         if p < m:
#             d.setdefault(x,p)
#     return d
def buid_dict(filename,arr,n,m):
    with open(filename, 'r') as f:
        ngram = ngrams_array(arr, n)
        for x in ngram:
            p = ngram.get(x)
            if p < m:
                f.write(x)

def build_sentence(input_arr):
    d = {}
    for x in range(len(input_arr)):
        d.setdefault(input_arr[x], x)
    chuoi = []
    for i in input_arr:
        x = d.get(i)
        if x == 0:
            chuoi.append(i)
        for j in input_arr:
            y = d.get(j)
            if y == x + 1:
                z = j.split(' ')
                chuoi.append(z[1])
    return " ".join(chuoi)

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

def load_keywords(filename):

    dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11,'12':12,'14':14,'15':15,'16':16,'17':17,'18':18,'19':19,'20':20,'21':21}
    array = []

    with open(filename,'r') as f:
        for line in f:
            line = line.rstrip('\n')

            label, key, quantity = line.split(" ", 3)
            try:
                array[dict[label]].append([key,quantity])
            except:
                print(label)

    return array
def load_data(filename, dict):
    res = []
    col1 = []; col2 = []; col3 = []; col4 = []
    dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11,
            '12': 12, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21}
    keywords = load_keywords("data/keyword.txt")
    with open(filename, 'r') as f,open(dict, "w") as f2:
        for line in f:
            if line !='\n':
                label_num, question = line.split("\t")
                # question = clean_str_vn(question)
                label, number = label_num.split("||#")
                # question = review_to_words(question,'dict/vietnamese-stopwords-dash.txt')
                # question = ViTokenizer.tokenize(unicode(question, encoding='utf-8'))
                question = accent(question)
                question = tokenizer.predict(question)
                question = regex_email(question)
                question = regex_phone_number(question)
                question = regex_link(question)
                array = question.split()

                col1.append(label)
                col2.append(number)
                for index in range(len(keywords[dict[label]])):

                    if keywords[dict[label]][index][0] in question:

                        for count in range((int)(keywords[dict[label]][index][1])):
                            array.append(keywords[dict[label]][index][0])
                question = ' '.join(array)
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
            q = review_to_words2(q,dict,2)  # q la 1 cau
            q1 = [' '.join(x) for x in ngrams(q, 1)]  # q1:mang cac 1-grams
            q2 = [' '.join(x) for x in ngrams(q, 2)]  # q2: mang cac phan tu 2-grams
            q3 = [' '.join(x.replace(' ', '_') for x in q2)]
            y = q1 + q3
            z = " ".join(y)
            col4.append(z)
            # col4.append(q)
        d = {"label":col1, "question": col4}
        train = pd.DataFrame(d)
        if filename == 'data/ques2.txt':
            joblib.dump(train,'model/train3.pkl')
        else:
            joblib.dump(train,'model/test3.pkl')
    return train

def training1():
    # vectorizer = load_model('model/vectorizer.pkl')
    # print "aa"
    # if vectorizer == None:
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    train = load_model('model/train3.pkl')
    if train == None:
        train = load_data('data/ques2.txt', 'dict/dict1')
    print "---------------------------"
    print "Training"
    print "---------------------------"
    train_text = train["question"].values
    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    X_train = X_train.toarray()
    y_train = train["label"]
    # print "train_data: \n", confusion_matrix(y_train, y_train,
    #                                              labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
    #                                                      "12", "14", "15", "16", "17", "18", "19", "20", "21"])
    joblib.dump(vectorizer, 'model/vectorizer3.pkl')
    fit1(X_train, y_train)

def fit1(X_train,y_train):
    uni_big = SVC(kernel='rbf', C=1000)
    uni_big.fit(X_train, y_train)
    joblib.dump(uni_big, 'model/uni_big3.pkl')

def predict_ex(mes):
    print mes
    uni_big = load_model('model/uni_big3.pkl')
    if uni_big == None:
        training1()
    uni_big = load_model('model/uni_big3.pkl')
    vectorizer = load_model('model/vectorizer3.pkl')
    t0 = time.time()
    # iterate over classifiers
    try:
        mes = unicode(mes, encoding='utf-8')
    except:
        mes = unicode(mes)
    mes = unicodedata.normalize("NFC",mes.strip())
    test_message = ViTokenizer.tokenize(mes).encode('utf8')
    test_message = clean_str_vn(test_message)
    test_message = list_words(test_message)
    clean_test_reviews = []
    clean_test_reviews.append(test_message)
    d2 = {"message": clean_test_reviews}
    test2 = pd.DataFrame(d2)
    test_text2 = test2["message"].values.astype('str')
    test_data_features = vectorizer.transform(test_text2)
    test_data_features = test_data_features.toarray()
    # print test_data_features
    s = uni_big.predict(test_data_features)[0]
    return s

def test_file():
    uni_big = load_model('model/uni_big3.pkl')
    if uni_big == None:
        training1()
    uni_big = load_model('model/uni_big3.pkl')
    vectorizer = load_model('model/vectorizer3.pkl')
    t0 = time.time()
    # iterate over classifiers
    test = load_model('model/test3.pkl')
    if test == None:
        test = load_data('data/test.txt','dict/dict1')
    test_text = test["question"].values
    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label"]
    y_pred = uni_big.predict(X_test)
    print y_pred

    print " accuracy: %0.3f" % f1_score(y_test, y_pred, average='weighted')
    # print "confuse matrix: \n", confusion_matrix(y_test, y_test,
    #                                              labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
    #                                                      "12", "14", "15", "16", "17", "18", "19", "20", "21"])

    # print "confuse matrix: \n", confusion_matrix(y_test, y_pred, labels=["0", "1", "2", "3", "4", "5","6","7","8","9","10", "11", "12", "14", "15","16","17","18","19","20","21"])

def test_mes():
    d = {"0": "Giá (sản phẩm, phụ kiện, thay linh kiện,...)", "1": "Cập nhật phần mềm hệ thống (cài phần mềm,...)",
         "2": "Tình trạng sp (còn hay hết)", "3": "So sánh 2 sản phẩm",
         "4": "Chế độ giao hàng", "5": "Chế độ bảo hành", "6": "Khuyến mãi", "7": "Thanh toán (tại cửa hàng, trả góp)",
         "8": "Trả góp",
         "9": "Chất lượng", "10": "Thông tin, chức năng", "11": "Báo khi có máy", "12": "Hủy đơn hàng", "14": "Đổi máy",
         "15": "Góp ý", "16": "Đánh giá tích cực", "17": "Phụ kiện (loại gì)", "18": "Khác", "19": "Tư vấn",
         "20": "Đặt trước máy",
         "21": "PMH có trừ trực tiếp vào máy ko"}
    mes = ""
    while(mes.lower()!="q"):
        mes = raw_input("Nhap cau: ")
        result = predict_ex(mes)
        print d[result]
if __name__ == '__main__':

    # mes = raw_input("Nhap cau: ")
    # test_mes()
    test_file()