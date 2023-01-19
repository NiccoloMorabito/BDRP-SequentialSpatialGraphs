import pandas as pd
from common_utils.stg_utils import CATEGORIES

pd.set_option('display.max_columns', None) 

def graph_to_dict(graph):
    return [node.__dict__ for node in graph.nodes] 

def dummy_categories(class_name_column):
    dtype = pd.CategoricalDtype(categories=CATEGORIES)
    cat = pd.Series(class_name_column, dtype=dtype)
    return pd.get_dummies(cat, prefix="class")

def graph_to_feature_vector(graph):
    df = pd.DataFrame.from_dict(graph_to_dict(graph)).set_index("id")
    df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"]) #TODO decide if you want to keep the area

    class_dummies = dummy_categories(df["class_name"])
    df = pd.concat([df,class_dummies], axis=1)
    df.drop('class_name', axis=1, inplace=True)

    df.drop('conf', axis=1, inplace=True) #TODO check if it's important, but it seems to be always 0.0
    df.drop('centroid', axis=1, inplace=True) #TODO check if it's important, but it's a tuple
    df.drop('detclass', axis=1, inplace=True) #TODO what is the meaning????
    #TODO you probably need to normalize (especially the coordinates and the area)
    return df.values