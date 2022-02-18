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

pcap_root_dir = './dataset'
features_output_dir = './features'
......
time_windows = [[0, 0.1], [0.1, 0.6], [0.6, 2.1], [2.1, 12.1], [12.1, 72.1]]

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

def get_pcap_paths_and_names(root_dir):
    # 递归得到所有pcap文件路径
    pcap_file_path_and_name = []
    for root, directory, files in os.walk(root_dir):  # 当前根, 根下目录, 目录下的文件
        for filename in files:
            name, suffix = os.path.splitext(filename)  # 文件名, 文件后缀
            if suffix in ['.pcap', '.pcapng']:
                pcap_file_path_and_name.append((os.path.join(root, filename), filename, "\\".join(root.split("\\")[1:])))
    return pcap_file_path_and_name


protocols_map = {}
protocols_len = len(protocols)

......

def features_extractor():
    global pcap_root_dir, features_output_dir, csv_dir, pre_dir_5, pre_dir_2
    pcap_root_dir = sys.argv[1]
    features_output_dir = sys.argv[2]
    csv_dir = os.path.join(features_output_dir, 'csv')
    ......

if __name__ == "__main__":
    features_extractor()