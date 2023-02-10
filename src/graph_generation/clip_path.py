# '/Users/xiaokeai/Documents/GitHub/yolov7-main/datasets/shanghaitech/testing/frames/01_0141'

# importing pandas
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
import pandas as pd
import networkx as nx
from scipy.spatial import distance
import pickle

root='data/yolo_annotated_datasets/StreetSceneDatasetResultsTesting_Label.txt'
# read text file into pandas DataFrame
df = pd.read_csv(root, sep=",",header=0)
# df['path']=df['path'].str.extract(r'^(.*[\\\/])')
df['path']=df['path'].str[:-1]

df.to_csv(r'data/yolo_annotated_datasets/StreetSceneDatasetResultsTesting_Label1.txt', header=True, index=False, sep=',', mode='w')