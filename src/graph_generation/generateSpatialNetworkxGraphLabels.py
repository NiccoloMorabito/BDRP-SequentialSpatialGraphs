# importing pandas
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
import pandas as pd
import networkx as nx
from scipy.spatial import distance
import pickle

# read text file into pandas DataFrame
df = pd.read_csv("datasets/AvenueDatasetResultsTesting_Label.txt", sep=",",header=0)
df_agg=df.groupby(['path']).agg(lambda x: x.tolist())
print(df_agg.index[5])
dic_label={}
for i in range(0,df_agg.shape[0]):
    dic_label[df_agg.index[i]]=df_agg.iloc[i,-1]

with open('results/AvenueDatasetResultsTestingLabels.pickle', 'wb') as handle:
    pickle.dump(dic_label, handle, protocol=pickle.HIGHEST_PROTOCOL)