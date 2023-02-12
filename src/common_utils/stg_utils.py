from os.path import join, exists
from typing import Union
from pathlib import Path
import cv2
from typing import Dict, Any, Union
from argparse import Namespace
import pickle
from dataclasses import dataclass, field

CATEGORIES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',\
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',\
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',\
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',\
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',\
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',\
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',\
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

EDGE_LABEL = "weight"

PARAMS_FILENAME = "params.pickle"

@dataclass#(unsafe_hash=True) #TODO removed the unsafe_hash to define a customized __hash__
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

    def __hash__(self):
        return hash(self.id)

    # TODO remove?
    def boundary_box(self) -> str:
        return f"({self.x1}, {self.y1}) - ({self.x2}, {self.y2})"

@dataclass(unsafe_hash=True)
class Edge:
    weight: Union[float, int] = field(default=0)

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

def save_video_params(source: str, dest: str, videoname: str): #TODO delete
    video_path = join(source, videoname)
    video_params = get_video_params(video_path)
    params_path = join(dest, videoname, PARAMS_FILENAME)
    with open(params_path, 'wb') as f:
        pickle.dump(video_params, f)

def load_video_params(video_folder: str) -> dict: #TODO delete
    params_path = join(video_folder, PARAMS_FILENAME)
    return load_pickle(params_path)

#TODO this function should not be in "stg_utils" -> change name of file?
def load_pickle(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)

def dict_with_attributes(d: Dict[str, Any]) -> Namespace:
    """
    Convert a dictionary to a class with attributes
    """
    return Namespace(**d)

#TODO delete
def print_log(text: str) -> None: print(f"[ log ] {text}")
def print_error(text: str) -> None: print(f"[ error ] {text}")