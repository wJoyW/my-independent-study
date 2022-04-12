import glob
import matplotlib.pyplot as plt
import os
import re
import time
import random
import numpy as np
import pickle
import math

from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_selection import SelectFromModel as skm
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import plot_confusion_matrix as pcm
from sklearn.neural_network import MLPClassifier as mlpc

from imblearn.ensemble import BalancedBaggingClassifier as bbc
from imblearn.datasets import make_imbalance

#用以計算耗時多久
def time_print(start, end, name):
    total = end - start
    minute = 0

    while total >= 60:
        minute += 1
        total -= 60

    print(f'In {name}, using {minute} minutes {total} seconds')

#取得log內容
def get_log(path):
    start_time = time.time()

    #路徑
    logs = glob.glob(path + '\*\*.log')
    log = []
    c = 0
    for g in logs:
        f = open(g, 'r', encoding = 'utf-8')

        #以檔案大小為門檻，太小之檔案不給予訓練或測試(單位：byte)
        if os.path.getsize(g) <= 1024:
            continue

        c += 1
        for l in f.readlines():
            log.append(l)
 
    end_time = time.time()
    print(f"Total files = {c}")
    time_print(start_time, end_time, "get_log")

    print(f'length of logs = {len(log)}')
    
    return log

#logs' preprocessing
def extract_feature(log):
    start_time = time.time()
    vocabulary = []
    vocab_in_file = []
    temp_vif = []

    ismal = []
    pdata = []
    cmd = ""
    process_attach = False
    pattern = re.compile('(\s)+=(\s)+(-)*(\d)+')
    ques_pattern = re.compile('(\s)+=(\s)+?')
    
    #若log中出現這些字元，則整筆不用
    ignore_list = ["No such process", "timeout", "Operation not permitted"
                   , "Invalid argument"]

    for line in log:
        if process_attach:
            ignore = False

            for i in ignore_list:
                if i in line and len(cmd) != 0:
                    ismal.pop(-1)
                    cmd = ""
                    temp_vif.clear()
                    process_attach = False
                    ignore = True
                    break
                    
            if not ignore:         
                if line.find('[SCAN_START]') != -1:
                    pdata.append(cmd)
                    vocab_in_file.append(temp_vif)
                    temp_vif.clear()
                    cmd = ""
                        
                    if line.split('MAL=')[1][0] == '0':
                        ismal.append(False)
                    else:
                        ismal.append(True)

                elif line.find("strace: Process") != -1 and line.find("detached") != -1:
                    pdata.append(cmd)
                    vocab_in_file.append(temp_vif)
                    temp_vif = []
                    cmd = ""
                    process_attach = False
                
                elif re.search(pattern, line) != None:
                    start = line.find(" ")
                    #end = line.find(re.search(pattern, line)[0])
                    end = line.find('(')
                    cmd += line[start : end]

                    if line[start : end] not in vocabulary:
                        vocabulary.append(line[start : end])

                    if line[start : end] not in temp_vif:
                        temp_vif.append(line[start : end])

                elif re.search(ques_pattern, line) != None:
                    start = line.find(" ")
                    end = line.find('(')
                    cmd += line[start : end]

                    if line[start : end] not in vocabulary:
                        vocabulary.append(line[start : end])

                    if line[start : end] not in temp_vif:
                        temp_vif.append(line[start : end])

                elif line.find("+++") != -1:
                    start = line.find("+++")
                    cmd += line[start:]
                    if line[start :] not in vocabulary:
                        vocabulary.append(line[start : end])

                    if line[start :] not in temp_vif:
                        temp_vif.append(line[start : end])

                    pdata.append(cmd)
                    vocab_in_file.append(temp_vif)
                    temp_vif.clear()
                    cmd = ""
                    process_attach = False

            ignore = False

        else:
            flag = False

            if line.find('[SCAN_START]') != -1:
                if line.find('MAL=') != -1:
                    if line.split('MAL=')[1][0] == '0':
                        ismal.append(0)
                    else:
                        ismal.append(1)
                    flag = True

                if flag:
                    process_attach = True
                else:
                    process_attach = False

    print(f'valid data = {len(pdata)}')
    
    if len(ismal) - len(pdata) == 1:
        pdata.append(cmd)
        vocab_in_file.append(temp_vif)
        temp_vif.clear()

    with open('vocab_in_file.pkl', 'wb') as f:
        pickle.dump(vocab_in_file, f)

    end_time = time.time()

    time_print(start_time, end_time, "extract_feature")
    print(f'benignware amount: {ismal.count(False)}')
    print(f'malware amount: {ismal.count(True)}')
    return pdata, ismal, vocabulary

#進行tfidf
def tfidf(data):
    start_time = time.time()

    counter = CountVectorizer(lowercase = False, vocabulary = vocabulary)
    vector = counter.fit(data)
    with open('vector.pkl', 'wb') as f:
        pickle.dump(vector, f)

    data = vector.transform(data)
    
    trans = TfidfTransformer()
    tfidf_machine = trans.fit(data)

    with open('tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf_machine, f)
    
    data = tfidf_machine.transform(data)

    #tfidf_machine = TfidfTransformer().fit(data)
    #with open('features.txt', 'w') as f:
    #    for t in tfidf_machine.get_feature_names():
    #        f.write(t)
    
    #data = tfidf_machine.transform(data)
    #with open('tfidf.pkl', 'wb') as f:
    #    pickle.dump(tfidf_machine, f)

    end_time = time.time()
    time_print(start_time, end_time, "tfidf")
    return data, vector, tfidf_machine

#進行cv，splits可調整cv次數
def model_cross(data, target, splits):
    start_time = time.time()

    target = np.array(target)

    skf = StratifiedKFold(n_splits = splits)
    count = 1
    
    for train_index, test_index in skf.split(data, target):
        xtrain, xtest = data[train_index], data[test_index]
        ytrain, ytest = target[train_index], target[test_index]

        #xtrain, xtest = tfidf(xtrain, xtest, count)

        clf = rfc(n_jobs = -1).fit(xtrain, ytrain)
        pred = clf.predict(xtest)
        f1 = f1_score(ytest, pred, average='binary')
        acc = accuracy_score(ytest, pred)
        
        print(f'f1 score = {f1}')
        print(f'accuracy = {acc}')
        #pcm(clf, xtest, ytest)
        #plt.savefig(f'{count}.png')

        with open(f'model{count}.pkl', 'wb') as f:
            pickle.dump(clf, f)
        count += 1

    end_time = time.time()
    time_print(start_time, end_time, "model_cross")


def main(path):
    log = get_log(path)
    pdata, ismal = extract_feature(log)
    tfidf_data = tfidf(pdata)
    model_cross(tfidf_data, ismal, 5)


if __name__ == '__main__':
    path = 'D:\大學\專業\專題\my專題\IoT_detection\IoT_detection\infected'
    #main(path)
    with open('./model1.pkl', 'rb') as f :
        model = pickle.load(f)
    model_select(model, path)