import pandas as pd
import numpy as np
import time
import pickle

from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import StratifiedKFold as skf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from imblearn.datasets import make_imbalance

start = time.time()

pd.set_option('display.max_columns', None)
df = pd.read_csv('F:\data_raw_14775.csv')
data = df.iloc[:, 2].to_numpy()
target = df.iloc[:, 1].to_numpy()
#print(data[0])

machine = TfidfVectorizer().fit(data)
with open('tf_machine.pkl', 'wb') as f:
    pickle.dump(machine, f)

tfidf = machine.transform(data)

amount = 60000
x_imb, y_imb = make_imbalance(tfidf, target, sampling_strategy = {0:amount, 1:amount})

cv = skf(n_splits = 5)
count = 1
for train, test in cv.split(x_imb, y_imb):
    xtrain, xtest = x_imb[train], x_imb[test]
    ytrain, ytest = y_imb[train], y_imb[test]

    clf = lr(n_jobs = -1).fit(xtrain, ytrain)
    with open(f'model{count}.pkl', 'wb') as f:
        pickle.dump(clf, f)

    count += 1
    pred = clf.predict(xtest)
    print(f1_score(pred, ytest))

end = time.time()
print(f'spend {end - start} seconds')