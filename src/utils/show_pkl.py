import os
import pickle

import sys

import joblib
import numpy as np


from hintsClasses.Hint import get_all_hints


script_path = os.path.dirname(os.path.abspath(__file__))

def show_pkl_content(pkl_file_path):
    """

    :param pkl_file_path: 要展示的pkl文件的位置
    :return:
    """
    hints = get_all_hints('mysql')
    data = joblib.load(pkl_file_path)
    print('数量: ' + str(len(data)))
    hint_ids = []
    key = True
    for d in data:
        # print(d)
        s, a, r, s2, done, hint_id = d
        # for hint in hints:
        #     if hint.hint_id == hint_id:
        #         des = hint.to_string()
        #         print(des + "reward : " + str(r))
        # hint_ids.append(hint_id)
        # print(hint_id)
    #print(data)

# show_pkl_content(demo_path)
