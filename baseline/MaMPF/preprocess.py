import tqdm
import numpy as np
import os
import sys
import json


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_origin_data(data_dir, app_num):
    datas = [[] for _ in range(app_num)]
    filenames = [filename for filename in os.listdir(data_dir) \
                 if os.path.isfile(os.path.join(data_dir, filename)) and filename.split(".")[1] == "json"]
    filenames = sorted(filenames);
    print(filenames);
    print(app_num);
    lable_names = {}
    for app in tqdm.tqdm(range(app_num), ascii=True, desc='[Load Data]'):
        lable_names[app] = filenames[app]
        with open(os.path.join(data_dir, filenames[app])) as fp:
            json_val = json.load(fp)
            flow_length = json_val['length']
            flow_status = json_val['type']
            for i in range(len(flow_length)):
                datas[app].append({'label': app, 'flow': flow_length[i], 'status': flow_status[i], 'lo': flow_length[i].copy()})
    data_json = json.dumps(lable_names)
    fileObject = open('/mnt/traffic/xzy/sp/MaMPF_1/lable_names.json', 'w')
    fileObject.write(data_json)
    fileObject.close()
    return datas


def _transform(datas, limit, max_packet):
    data_trans = [[] for _ in range(len(datas))]
    for app in tqdm.tqdm(range(len(datas)), ascii=True, desc='[Transform]'):
        app_data = datas[app]
        for idx, example in enumerate(app_data):
            flow = example['flow']
            if len(flow) < limit:
                print(flow)
                continue
            flow = [ix if ix <= max_packet else max_packet for ix in flow]
            data_trans[app].append(
                {'label': example['label'], 'flow': flow, 'status': example['status'], 'lo': example['lo'],
                 'id': str(app) + '-' + str(idx)}
            )
    return data_trans


def split_train_and_dev(datas, ratio=0.8, keep_ratio=1):
    train, dev = [], []
    for app_data in tqdm.tqdm(datas, ascii=True, desc='[Split]'):
        is_keep = np.random.rand(len(app_data)) <= keep_ratio
        is_train = np.random.rand(len(app_data)) <= ratio
        for example, kp, tr in zip(app_data, is_keep, is_train):
            if kp and tr:
                train.append(example)
            elif kp and not tr:
                dev.append(example)
    np.random.shuffle(train)
    np.random.shuffle(dev)
    return train, dev


def preprocess(config):
    eprint('Generate train and test.')
    origin = load_origin_data(config.data_dir, config.class_num)
    length = _transform(origin, config.min_length, config.max_packet_length)
    train, test = split_train_and_dev(length, config.split_ratio, config.keep_ratio)
    with open(config.train_json, 'w') as fp:
        json.dump(train, fp, indent=1)
    with open(config.test_json, 'w') as fp:
        json.dump(test, fp, indent=1)

