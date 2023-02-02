import cv2
import os.path
from typing import Optional, Tuple, Union
import pickle
import glob
# importing pandas
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
import pandas as pd
import networkx as nx
from scipy.spatial import distance
import pickle

# read text file into pandas DataFrame
df = pd.read_csv("results/AvenueDatasetResults.txt", sep=",",header=0)
dictmap=dict()

list_nameMapping=[]
list_nameMapping=list(zip(df.iloc[:,0], df['video_no']))
list_nameMapping=list(dict.fromkeys(list_nameMapping))
print(list_nameMapping)

video_folder='/datasets/Avenue Dataset'
video_folder=os.getcwd()+video_folder

params_path='video_parameters/video_parameters_AvenueDatasets.pickle'
def get_video_params(video_path) -> dict:
    if not os.path.exists(video_path): raise FileNotFoundError("Video file not found")
    cap = cv2.VideoCapture(video_path)
    params = dict({
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "nframes": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    })
    cap.release()
    return params

videoname2params = dict()


for ext in ('**/*.mov', '**/*.avi', '**/*.mp4','**/*.mpg', '**/*.mpeg', '**/*.m4v','**/*.wmv', '**/*.mkv','**/*.bmp', '**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.tif', '**/*.tiff', '**/*.dng', '**/*.webp', '**/*.mpo'):
    for video in (glob.iglob(os.path.join(video_folder, ext), recursive=True)):
        video_path = os.path.join(video_folder, video)
        for i in range(0,len(list_nameMapping)):
            if video_path==list_nameMapping[i][0]:
                params = get_video_params(video_path)
                videoname2params['video_'+str(list_nameMapping[i][1])]= params


with open(params_path, 'wb') as f:
        pickle.dump(videoname2params, f)

with open(params_path, 'rb') as handle:
    dictAvenue = pickle.load(handle)
    print(dictAvenue)

    