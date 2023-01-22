from os.path import join, exists
from typing import Union
from pathlib import Path
import cv2
from typing import List, Tuple, Dict, Any, Union
from argparse import Namespace
import pickle

CATEGORIES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',\
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',\
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',\
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',\
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',\
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',\
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',\
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

#TODO check the value of these contants (change also names of the variables)
STG_SPATIAL_DIST_LABEL = "weight"
SG_SPATIAL_DIST_LABEL = "distance"
SG_SPATIAL_SPEED_LABEL = "speed"

PARAMS_FILENAME = "params.pickle"

def get_video_params(video_path: Union[Path, str]) -> dict:
    if not exists(video_path): raise FileNotFoundError("Video file not found")
    cap = cv2.VideoCapture(video_path)
    params = dict({
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "nframes": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    })
    cap.release()
    return params

def save_video_params(source: str, dest: str, videoname: str):
    video_path = join(source, videoname)
    video_params = get_video_params(video_path)
    params_path = join(dest, videoname, PARAMS_FILENAME)
    with open(params_path, 'wb') as f:
        pickle.dump(video_params, f)

def load_video_params(video_folder: str) -> dict:
    params_path = join(video_folder, PARAMS_FILENAME)
    with open(params_path, 'rb') as f:
        return pickle.load(f)

def dict_with_attributes(d: Dict[str, Any]) -> Namespace:
    """
    Convert a dictionary to a class with attributes
    """
    return Namespace(**d)

#TODO delete
def print_log(text: str) -> None: print(f"[ log ] {text}")
def print_error(text: str) -> None: print(f"[ error ] {text}")