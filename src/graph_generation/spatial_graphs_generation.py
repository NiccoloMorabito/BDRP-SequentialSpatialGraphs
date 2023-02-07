import pandas as pd
import networkx as nx
from scipy.spatial import distance
import pickle
import os

import sys
sys.path.append("..")

from common_utils.stg_utils import Node, get_video_params

CLASSES_LIST =  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear']

VIDEOS_FOLDER           = "" #TODO add path to the videos folder
TXT_FOLDER              = "../../data/yolo_annotated_datasets/"
TRAINING_GRAPH_FOLDER   = "../../data/training_graphs/"
TESTING_GRAPH_FOLDER    = "../../data/testing_graphs/"
VIDEOPARAMS_FOLDER      = "../../data/video_parameters/"
TESTING_LABELS_FOLDER   = "../../data/testing_labels/"

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
    for i in df.video_no.unique():
        dictAvenue[str(df[df['video_no']==i]['path_org'].values[0])]=[]
        for j in range(df[df['video_no']==i]['frame_no'].min(),df[df['video_no']==i]['frame_no'].max()):
            graph = nx.Graph()
            df_new=df[(df['video_no']==i) & (df['frame_no']==j)]
            for row in df_new.iterrows():
                graph.add_node(Node(row[1][5],row[1][6],row[1][7],row[1][8],row[1][9],row[1][10],row[1][11],row[1][13],row[1][14]))
            for node1 in graph.nodes:
                for node2 in graph.nodes:
                    if node1.id == node2.id: continue
                    graph.add_edge(node1, node2, weight=distance.euclidean(node1.centroid, node2.centroid))
            dictAvenue[str(df[df['video_no']==i]['path_org'].values[0])].append(graph)

    #  save graph dict
    with open(pickle_path, 'wb') as handle:
        pickle.dump(dictAvenue, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_videoparams(dataset_folder, params_pickle_path):
    videoname2params = dict()

    for video in os.listdir(dataset_folder):
        video_path = os.path.join(dataset_folder, video)
        params = get_video_params(video_path)
        videoname2params[video_path] = params

    with open(params_pickle_path, 'wb') as f:
        pickle.dump(videoname2params, f)

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

    dataset_name = "AvenueDataset" #TODO change according to the dataset

    ''' FOR TRAINING SET '''
    txt_path = os.path.join(TXT_FOLDER, f"{dataset_name}ResultsTraining.txt")
    graphs_pickle_path = os.path.join(TRAINING_GRAPH_FOLDER, f"{dataset_name}_training.pickle")
    generate_spatial_graph_from_txt(txt_path, graphs_pickle_path)

    videos_folder = os.path.join(VIDEOS_FOLDER, dataset_name)
    params_pickle_path = os.path.join(VIDEOPARAMS_FOLDER, f"{dataset_name}_video_params.pickle")
    generate_videoparams(videos_folder, params_pickle_path)

    ''' FOR TESTING SET '''
    txt_path = os.path.join(TXT_FOLDER, f"{dataset_name}ResultsTesting.txt")
    graphs_pickle_path = os.path.join(TESTING_GRAPH_FOLDER, f"{dataset_name}_testing.pickle")
    generate_spatial_graph_from_txt(txt_path, graphs_pickle_path)

    labels_txt_path = os.path.join(TXT_FOLDER, f"{dataset_name}ResultsTesting_Label.txt")
    labels_pickle_path = os.path.join(TESTING_LABELS_FOLDER, f"{dataset_name}_testing_labels.pickle")
    generate_labels_from_txt(labels_txt_path, labels_pickle_path)
