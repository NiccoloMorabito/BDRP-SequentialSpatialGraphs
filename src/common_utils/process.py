import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch

from common_utils.stg_utils import CATEGORIES

pd.set_option('display.max_columns', None) 

def graph_to_feature_vector(graph, normalize: bool = False): # normalization seems to affect negatively the performance
    df = pd.DataFrame.from_dict(graph_to_dict(graph)).set_index("id")
    df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])

    # one-hot encoding of categorical feature
    class_dummies = dummy_categories(df["class_name"]) #TODO reduce the number of classes
    df = pd.concat([df,class_dummies], axis=1)
    df.drop('class_name', axis=1, inplace=True)

    # removing useless features
    df.drop('conf', axis=1, inplace=True) # removed because it seems to be always 0.0
    df.drop('centroid', axis=1, inplace=True)
    df.drop('detclass', axis=1, inplace=True)
    
    feature_vector =  torch.FloatTensor(df.values)
    if normalize:
        return torch.nn.functional.normalize(feature_vector)
    return feature_vector

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