# coding:utf-8
import os
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
import numpy as np
from .models import *
import json
#进度条...
import tqdm
import eval
import traceback
import time;

_MODEL = {
	#论文..
    'FoSM': SMarkovModel(order=1), #MTS
    'SoSM': SMarkovModel(order=2),
    'FoLM': LMarkovModel(order=1), #LBS
    'SLC-LR': SLMarkovClassify(LR(C=1, class_weight=None), 1),
    'SLC-RF': SLMarkovClassify(RF(n_estimators=50), 1),
    'SOB-L':SMarkovModelWithCluster(n_cluster=40, order=2),
    'SOCRT-L':SMarkovModelWithCluster(n_cluster=5, order=2),
    'SOCRT': SMarkovModelWithCluster(n_cluster=5, order=2),  # , 5
    'SOB': SMarkovModelWithCluster(n_cluster=40, order=2),  # 40, 110(best)
}


_CERT_KEY_BASE = '22:11'
_START_KEY_BASE = '23:-2'

#读取数据...
def _data_read_from_json(files, app_num):
    status = [[] for _ in range(app_num)]
    lengths = [[] for _ in range(app_num)]
    with open(files) as fp:
        data = json.load(fp)
    for exp in tqdm.tqdm(data, ascii=True, desc='[Read]'):
        app = exp['label']
        status[app].append(exp['status'])
        lengths[app].append(exp['lo'])
    return status, lengths

#读取key_set...
def _get_key_set(label_file, seek_key=(_CERT_KEY_BASE, _START_KEY_BASE)):
    all_keys = [set() for _ in range(len(seek_key))]
    with open(label_file) as fp:
        label = json.load(fp)
    for kx in label.keys():
        for ik, sk in enumerate(seek_key):
            if sk in kx:
                all_keys[ik].add(label[kx])
    return all_keys

#不知道在干啥....
def _data_key_length(status, length, key_set):
    key_packet_length = []
    for app_s, app_l in zip(status, length):
        app_kpl = []
        for sf, lf in zip(app_s, app_l):
            kpl = 0
            for ps, pl in zip(sf, lf):
                if ps in key_set:
                    kpl = pl
                    break
            app_kpl.append(kpl)
        key_packet_length.append(app_kpl)
    return key_packet_length

#获取标签...
def _get_label(test):
    label = []
    for app_ind, app_data in enumerate(test):
        label.append(np.ones(len(app_data)) * app_ind)
    return label

#额...
def get_data(config):
    train_s, train_l = _data_read_from_json(config.train_json, config.class_num)
    test_s, test_l = _data_read_from_json(config.test_json, config.class_num)
    test_label = _get_label(test_s)

    train_cert = None;
    test_cert = None;

    train_start = None;
    test_start = None;

    with open(config.train_two_record) as fp:
        data = json.load(fp);
        train_cert = data['cert_set'];
        train_start = data['start_set'];
    with open(config.test_two_record) as fp:
        data = json.load(fp);
        test_cert = data['cert_set'];
        test_start = data['start_set'];
    print(len(test_s));
    print(len(test_label));

    return (train_s, train_l, None, train_cert, train_start), \
           (test_s, test_l, test_label, test_cert, test_start)
    # return (train_s, train_l, None, None, None), \
    #        (test_s, test_l, test_label, None, None);

#额...
def _combine_key_packet_length(*key_packt_length):
    combined = []
    for app_data in zip(*key_packt_length):
        combined.append(list(zip(*app_data)))
    return combined

#额...
def _check_model(model_names):
    for ix in model_names:
        if ix not in _MODEL:
            raise ValueError('Do not have the model %s' % ix)


def markov(config):
    model_names = [ix for ix in str(config.markov_models).split('#') if ix != '']
    print(model_names)
    _check_model(model_names)
    train, test = get_data(config);
    res_all = {}
    for mx in tqdm.tqdm(model_names, desc='Model', ascii=True):
        modx = _MODEL[mx]
        try:
            if mx == 'SOCRT':
                time1 = time.time();
                modx.fit(train[0], train[3])
                time2 = time.time();
                pred = modx.predict(test[0], test[3]);
                time3 = time.time();
            elif mx == 'SOCRT-L':
                time1 = time.time();
                modx.fit(train[1], train[3])
                time2 = time.time();
                pred = modx.predict(test[1], test[3]);
                time3 = time.time();
            elif mx == 'SOB':
                time1 = time.time();
                modx.fit(train[0], _combine_key_packet_length(train[3], train[4]))
                time2 = time.time();
                pred = modx.predict(test[0], _combine_key_packet_length(test[3], test[4]))
                time3 = time.time();
            elif mx == 'SOB-L':
                time1 = time.time();
                modx.fit(train[1], _combine_key_packet_length(train[3], train[4]))
                time2 = time.time();
                pred = modx.predict(test[1], _combine_key_packet_length(test[3], test[4]))
                time3 = time.time();
            #SMarkovModel order=1
            elif mx == 'FoLM':
                time1 = time.time();
                modx.fit(train[1], train[0])
                time2 = time.time();
                pred = modx.predict(test[1], test[0])
                time3 = time.time();
            else:
                time1 = time.time();
                modx.fit(train[0], train[1])
                time2 = time.time();
                pred = modx.predict(test[0], test[1])
                time3 = time.time();

            fp = open(mx+'_time.txt','w');
            fp.write('训练时间:' + str(time2 - time1)+'\n');
            fp.write('测试时间:' + str(time3 - time2)+'\n');
            fp.close();

            res = eval.evaluate(test[2], pred)
            eval.save_res(res, os.path.join(config.pred_dir, str(mx) + '.json'))
            res_all[mx] = res
            p_res = {kx: vx for kx, vx in res.items() if kx in ['FTF', 'ACC', 'AVE-TPR', 'AVE-FPR']}
            tqdm.tqdm.write('[RESULT] {}: {}'.format(mx, json.dumps(p_res, sort_keys=True, ensure_ascii=False)))
        except Exception as e:
            print('[ERROR] model `{}`'.format(mx))
            traceback.print_exc()
