# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import csv
import io
import sys
import random
import json
import traceback
import time

from sklearn import metrics
from sklearn import svm
from sklearn import cluster
import sklearn
import csv
import pandas as pd
import argparse
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn import tree

......

save_model_dir = None

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def RF(trainX,trainY,testX,testY):
    rfc = RandomForestClassifier(n_estimators=50, class_weight='balanced')
    rfc.fit(trainX, trainY)
    
    if save_model_dir != None:
        joblib.dump(rfc, os.path.join(save_model_dir, 'RF.pkl'))

    predict_result = rfc.predict(testX)
    acc = metrics.accuracy_score(testY,predict_result)
    recall = metrics.recall_score(testY, predict_result, average="weighted") #macro
    precision = metrics.precision_score(testY, predict_result, average="weighted")
    F1 = metrics.f1_score(testY,predict_result,average="weighted") #weighted
    return acc,recall,precision,F1

def DF(trainX,trainY,testX,testY):
    dfc = tree.DecisionTreeClassifier()
    dfc.fit(trainX,trainY)

    if save_model_dir != None:
        joblib.dump(dfc, os.path.join(save_model_dir, 'DF.pkl'))
    
    predict_result = dfc.predict(testX)
    acc = metrics.accuracy_score(testY,predict_result)
    recall = metrics.recall_score(testY, predict_result, average="weighted")
    precision = metrics.precision_score(testY, predict_result, average="weighted")
    F1 = metrics.f1_score(testY,predict_result,average="weighted")
    return acc,recall,precision,F1

def XGB(trainX,trainY,testX,testY):
    ......

def detect():
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    model = sys.argv[3]

    mode = sys.argv[4]

    global save_model_dir
    if len(sys.argv) >= 6:
        save_model_dir = sys.argv[5]
    
    if mode == 'train':
        ......
    elif mode == 'test':
        ......

if __name__ == "__main__":
    detect()