# importing pandas
import pandas as pd
import networkx as nx
from scipy.spatial import distance
import pickle

from common_utils.stg_utils import Node

CLASSES_LIST =  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear']

def generate_spatial_graph_from_txt(txt_path, pickle_path):
    # read text file into pandas DataFrame
    df = pd.read_csv(txt_path, sep=",",header=0)

    # preprocessing the df
    indexlist=[*range(0, 78, 1)]
    mapdict = {indexlist[i]: CLASSES_LIST[i] for i in range(len(indexlist))}

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
    with open(pickle_path, 'wb') as handle:
        pickle.dump(dictAvenue, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    #  load graph dict
    with open(pickle_path, 'rb') as handle:
        dictAvenue = pickle.load(handle)
        print(dictAvenue)
    '''

def generate_labels_from_txt(txt_path, pickle_path):
    # read text file into pandas DataFrame
    df = pd.read_csv(txt_path, sep=",",header=0)
    df_agg=df.groupby(['path']).agg(lambda x: x.tolist())
    print(df_agg.index[5])
    dic_label={}
    for i in range(0,df_agg.shape[0]):
        dic_label[df_agg.index[i]]=df_agg.iloc[i,-1]

    with open(pickle_path, 'wb') as handle:
        pickle.dump(dic_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    '''
    txt_path = "datasets/AvenueDatasetResultsTesting.txt"
    pickle_path = "results/AvenueDatasetResultsTesting.pickle"
    generate_spatial_graph_from_txt(txt_path, pickle_path)
    '''

    labels_txt_path = "datasets/AvenueDatasetResultsTesting_Label.txt"
    labels_pickle_path = "results/AvenueDatasetResultsTestingLabels.pickle"
    generate_labels_from_txt()