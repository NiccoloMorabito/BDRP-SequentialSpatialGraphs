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

root="data/yolo_annotated_datasets/StreetSceneDatasetResultsTesting_Label.txt"
df=pd.read_csv(root, sep=",",header=0)
df_label_list=df.groupby('path')['abnormal'].apply(list)
dic_label={}
for i in range(len(df_label_list.keys())):
    dic_label[df_label_list.keys()[i]]=list(df_label_list[i])

with open('data/testing_labels/StreetScene_testing_labels.pickle', 'wb') as handle:
    pickle.dump(dic_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/testing_labels/StreetScene_testing_labels.pickle', 'rb') as handle:
     dictAvenue = pickle.load(handle)
     print(dictAvenue)

