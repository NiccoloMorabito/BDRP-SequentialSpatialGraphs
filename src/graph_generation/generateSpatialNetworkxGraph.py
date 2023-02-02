# importing pandas
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
import pandas as pd
import networkx as nx
from scipy.spatial import distance
import pickle

# read text file into pandas DataFrame
df = pd.read_csv("datasets/AvenueDatasetResultsTesting.txt", sep=",",header=0)

# preprocessing the df
namelist=  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear']
indexlist=[*range(0, 78, 1)]
mapdict = {indexlist[i]: namelist[i] for i in range(len(indexlist))}


df['path_org'] = df['path']
df['path'] = df['path'].str.extract(r'(\d+)(?!.*\d)')
df['path'].astype(int)
df['video_no'].astype(int)
df['x1'].astype(float)
df['x2'].astype(float)
df['y1'].astype(float)
df['y2'].astype(float)
df['class_name']=df['detclass']
df['class_name']=df.class_name.map(mapdict)
df['centroid']= list(zip((df['x1'] + df['x2'])*0.5, (df['y1'] + df['y2'])*0.5))

#create node and edge class
@dataclass(unsafe_hash=True)
class Node:
    id: int = field(default=0)
    x1: int = field(default=0)
    y1: int = field(default=0)
    x2: int = field(default=0)
    y2: int = field(default=0)
    conf: float = field(default=float(0))
    detclass: int = field(default=0)
    class_name: str = field(default="")
    centroid: tuple = field(default=(0, 0))

@dataclass(unsafe_hash=True)
class Edge:
    weight: Union[float, int] = field(default=0)


list_nameMapping=[]
list_nameMapping=list(zip(df['path_org'], df['video_no']))
list_nameMapping=list(dict.fromkeys(list_nameMapping))

# feed data into the graph
dictAvenue={} # create a dict to save all videos, each video (dict key) is a list (dict value) of frames, each frame is a nx graph
for i in range(df['video_no'].min(),df['video_no'].max()+1 ):
    if i==list_nameMapping[i-df['video_no'].min()][1]:
            dictAvenue[list_nameMapping[i-df['video_no'].min()][0]]=[]
    for j in range(df[df['video_no']==i]['frame_no'].min(),df[df['video_no']==i]['frame_no'].max()):
        graph = nx.Graph()
        df_new=df[(df['video_no']==i) & (df['frame_no']==j)]
        for row in df_new.iterrows():
            graph.add_node(Node(row[1][5],row[1][6],row[1][7],row[1][8],row[1][9],row[1][10],row[1][11],row[1][13],row[1][14]))
        for node1 in graph.nodes:
            for node2 in graph.nodes:
                if node1.id == node2.id: continue
                graph.add_edge(node1, node2, weight=distance.euclidean(node1.centroid, node2.centroid))
        dictAvenue[list_nameMapping[i-df['video_no'].min()][0]].append(graph)
        # dictAvenue['video_'+str(i)].append(graph)

#  save graph dict
with open('results/AvenueDatasetResultsTesting.pickle', 'wb') as handle:
    pickle.dump(dictAvenue, handle, protocol=pickle.HIGHEST_PROTOCOL)

#  load graph dict
with open('results/AvenueDatasetResultsTesting.pickle', 'rb') as handle:
    dictAvenue = pickle.load(handle)
    print(dictAvenue)