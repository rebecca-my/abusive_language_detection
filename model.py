import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

basepath = 'your_path'

train_raw = basepath + 'cleantrain.txt'
test_raw = basepath + 'cleantest.txt'
train_gold_raw = basepath + 'waseemtrainGold.txt'
test_gold_raw = basepath + 'waseemtestGold.txt'

with open(train_raw) as f:
    train = f.readlines()
with open(test_raw) as f:
    test = f.readlines()
with open(train_gold_raw) as f:
    train_gold = f.readlines()
    train_gold = [int(s.strip()) for s in train_gold]
with open(test_gold_raw) as f:
    test_gold = f.readlines()
    test_gold = [int(s.strip()) for s in test_gold]

scaler = Vectorizer(analyzer='char', ngram_range=(1,3), strip_accents='ascii')
scaler.fit(train)
train_vecs = scaler.transform(train)
test_vecs = scaler.transform(test)

for C in [1e-3, 1e-2, 1e-1, 1e0, 1e1]:
    model_lr = LogisticRegression(C=C, solver='saga', penalty='l1')
    model_lr.fit(train_vecs, train_gold)
    pred = model_lr.predict(test_vecs)
    print(C)
    print(classification_report(test_gold, pred))

tokens = np.array(scaler.get_feature_names())

## top 10 offensive 3-grams
tokens[np.argsort(model_lr.coef_)[0,:10]].tolist()

## top 10 most predictive 3-grams
list(reversed(tokens[np.argsort(np.abs(model_lr.coef_))[0,-10:]]))