import numpy
import os
import glob
import re 
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
import pandas as pd
import networkx as nx
from scipy.spatial import distance
import pickle


list_key=[]
list_value=[]

for file in glob.glob(r"/Users/xiaokeai/Documents/GitHub/yolov7-main/datasets/shanghaitech/testing/test_frame_mask/*.npy"):
    try:
        a=numpy.load(file,allow_pickle=True)
        # print(len(a))
        # print(str(file))
        list_key.append(str(file))
        list_value.append(a)
    except:
        pass 

list_key_replace = [s.replace('/Users/xiaokeai/Documents/GitHub/yolov7-main/datasets/shanghaitech/testing/test_frame_mask/', '/Users/xiaokeai/Documents/GitHub/yolov7-main/datasets/shanghaitech/testing/frames/') for s in list_key]
list_key_replace = [s.replace('.npy', '') for s in list_key_replace]

dic_label={}
for i in range(len(list_key_replace)):
    dic_label[list_key_replace[i]]=list(list_value[i])

with open('results/ShanghaiTech_testing_labels.pickle', 'wb') as handle:
    pickle.dump(dic_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('results/ShanghaiTech_testing_labels.pickle', 'rb') as handle:
     dictAvenue = pickle.load(handle)
     print(dictAvenue.keys())

