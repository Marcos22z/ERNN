from datetime import datetime
import sys
sys.path.append('..') 

import torch as th
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import json
import os

def precision_k(pred, label, k=[1, 3, 5]):
    batch_size = pred.shape[0]
    
    precision = []
    for _k in k:
        p = 0
        for i in range(batch_size):
            p += label[i, pred[i, :_k]].mean().item()
        precision.append(p*100/batch_size)
    
    return precision


def ndcg_k(pred, label, k=[1, 3, 5]):
    batch_size = pred.shape[0]
    
    ndcg = []
    for _k in k:
        score = 0
        rank = th.log2(th.arange(2, 2 + _k, dtype=label.dtype, device=label.device))
        for i in range(batch_size):
            l = label[i, pred[i, :_k]]
            n = l.sum().item()
            if(n == 0):
                continue
            
            dcg = (l/rank).sum().item()
            label_count = label[i].sum().item()
            norm = 1 / th.log2(th.arange(2, 2 + min(_k, label_count), dtype=label.dtype))
            norm = norm.sum().item()
            score += dcg/norm
            
        ndcg.append(score*100/batch_size)
    
    return ndcg

def eve(real, pred):
    acc = accuracy_score(real, pred)
    f1 = f1_score(real, pred, average = 'macro')
    return acc, f1

def evaluate(net, if_log=False, test_data_loader=None, data_path='./data/sample', test_batch_size=50, word_num=500):
    if(test_data_loader == None):
        test_data_loader = load_test_data(data_path, test_batch_size, word_num)
    
    p1, p3, p5 = 0, 0, 0
    ndcg1, ndcg3, ndcg5 = 0, 0, 0
    
    pred_values = []
    label_values = []

    # number = 1
    with th.no_grad():
        for batch_idx, (X_batch, y_batch) in tqdm(enumerate(test_data_loader), desc='evaluating'):
            # if number >= 10:
            #     break
            # number = number + 1
            _batch_size = X_batch.shape[0]
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()

            output = net(X_batch)
            pred = output.topk(k=5)[1]
            
            batch_size = pred.shape[0]
            for i in range(batch_size):
                pred_values.append(int(pred[i,0]))
                label_values.append(int(y_batch[i].argmax()))

            # print("pred=" + pred[0])

            _p1, _p3, _p5 = precision_k(pred, y_batch, k=[1, 3, 5])
            p1 += _p1
            p3 += _p3
            p5 += _p5

            _ndcg1, _ndcg3, _ndcg5 = ndcg_k(pred, y_batch, k=[1, 3, 5])
            ndcg1 += _ndcg1
            ndcg3 += _ndcg3
            ndcg5 += _ndcg5
    
    batch_idx += 1
    p1 /= batch_idx
    p3 /= batch_idx
    p5 /= batch_idx
    ndcg1 /= batch_idx
    ndcg3 /= batch_idx
    ndcg5 /= batch_idx
    
    print('P@1\t%.3f\t\tP@3\t%.3f\t\tP@5\t%.3f' %(p1, p3, p5))
    print('nDCG@1\t%.3f\t\tnDCG@3\t%.3f\t\tnDCG@5\t%.3f' %(ndcg1, ndcg3, ndcg5))
    print(label_values)
    print(pred_values)
    oa, f1 = eve(label_values, pred_values)
    
    print('OA='+str(oa) + ',f1=' + str(f1))
    number = 0
    for i in range(len(label_values)):
        if label_values[i] == pred_values[i]:
            number = number + 1
    # data = {}
    # data['pred_values'] = pred_values
    # data['label_values'] = label_values
    # data_json = json.dumps(data);
    # fileObject = open(os.path.join(data_path, 'result.json'), 'w')
    # fileObject.write(data_json)
    # fileObject.close()
    # print(".json文件输出至目的！")

    print('precision='+str(number/len(label_values)))
    
    if(if_log):
        log_columns = ['P@1', 'P@3', 'P@5', 'nDCGP@1', 'nDCG@3', 'nDCG@5', 'OA', 'F1', 'PRE']
        log = pd.DataFrame([[p1, p3, p5, ndcg1, ndcg3, ndcg5, oa, f1, number/len(label_values)]], columns=log_columns)
        log.to_csv('./log/result-' + str(datetime.now()) + '.csv', encoding='utf-8', index=False)
        # log_columns = ['P@1', 'P@3', 'P@5', 'nDCGP@1', 'nDCG@3', 'nDCG@5']
        # log = pd.DataFrame([[p1, p3, p5, ndcg1, ndcg3, ndcg5]], columns=log_columns)
        # log.to_csv('./log/result-' + str(datetime.now()) + '.csv', encoding='utf-8', index=False)