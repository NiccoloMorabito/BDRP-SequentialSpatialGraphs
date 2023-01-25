import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch
import networkx as nx

from common_utils.stg_utils import CATEGORIES

pd.set_option('display.max_columns', None) 

def graph_to_feature_vector(graph):
    df = pd.DataFrame.from_dict(graph_to_dict(graph)).set_index("id")
    df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"]) #TODO decide if you want to keep the area

    class_dummies = dummy_categories(df["class_name"]) #TODO reduce the number of classes
    df = pd.concat([df,class_dummies], axis=1)
    df.drop('class_name', axis=1, inplace=True)

    df.drop('conf', axis=1, inplace=True) #TODO check if it's important, but it seems to be always 0.0
    df.drop('centroid', axis=1, inplace=True) #TODO check if it's important, but it's a tuple
    df.drop('detclass', axis=1, inplace=True) #TODO what is the meaning????
    #TODO you probably need to normalize (especially the coordinates and the area)
    return torch.FloatTensor(df.values)

def graph_to_dict(graph):
    return [node.__dict__ for node in graph.nodes] 

def dummy_categories(class_name_column):
    dtype = pd.CategoricalDtype(categories=CATEGORIES)
    cat = pd.Series(class_name_column, dtype=dtype)
    return pd.get_dummies(cat, prefix="class")



def adj_to_normalized_tensor(adj):
    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj = torch.FloatTensor(adj)
    return adj

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()