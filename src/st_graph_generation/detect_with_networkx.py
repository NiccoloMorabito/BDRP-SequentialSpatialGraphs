import time
from pathlib import Path
from os.path import join
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from sys import path
from scipy.spatial import distance
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
import networkx as nx

YOLO_PATH = "/Users/niccolomorabito/morabito.1808746@studenti.uniroma1.it - Google Drive/My Drive/BDMA/Semester3 CS/Big Data Research Project/Big-Data-Research-Project/src/object_detection/yolov7_with_object_tracking"    
MODEL_PATH = join(YOLO_PATH, "yolov7.pt")

#TODO prob remove
SEED = 0 # random seed to control bbox colors ?
np.random.seed(SEED)
THICKNESS = 2 # bounding box and font size thickness
NOBBOX = False # don't show bounding box
NOLABEL = False # don't show label

path.append(YOLO_PATH)
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from sort import *

sort_tracker = Sort(max_age=5,
                    min_hits=2,
                    iou_threshold=0.2)

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

    # TODO remove?
    def boundary_box(self) -> str:
        return f"tl: ({self.x1}, {self.y1}) - br: ({self.x2}, {self.y2})"

@dataclass(unsafe_hash=True)
class Edge:
    weight: Union[float, int] = field(default=0)

def draw_boxes(img, bbox, identities=None, categories=None, confidences = None, names=None, colors = None):
    """
    Function to Draw Bounding boxes
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = THICKNESS or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat]
        
        if not NOBBOX:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        if not NOBBOX:
            label = str(id) + ":"+ names[cat] if identities is not None else  f'{names[cat]} {confidences[i]:.2f}'
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def generate_spatial_graph(img, bbox, identities=None, categories=None, confidences = None, names=None, colors = None):
    """
    Construct a spatial graph from the bounding boxes, identities, categories, confidences, names and colors
    """
    graph = nx.Graph()
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        conf = confidences[i] if confidences is not None else 0
        class_name = names[cat]

        graph.add_node(Node(id, x1, y1, x2, y2, conf, cat, class_name, centroid))

        tl = THICKNESS or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        color = colors[cat]
        
        if not NOBBOX:
            cv2.circle(img, centroid, 5 * tl, color, tl)

        if not NOLABEL:
            label = f"Node({str(id)}): {names[cat]} {confidences[i]:.2f}" if identities is not None else  f'{names[cat]} {confidences[i]:.2f}'
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.circle(img, centroid, tf * 5, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    line_color = [0, 0, 0]
    line_thickness = 1
    # Add edges to the graph
    for node1 in graph.nodes:
        for node2 in graph.nodes:
            if node1.id == node2.id: continue
            # TODO EDGE CREATION MUST BE DIFFERENT
            graph.add_edge(node1, node2, weight=distance.euclidean(node1.centroid, node2.centroid))
            cv2.line(img, node1.centroid, node2.centroid, line_color, line_thickness)

    return img, graph

def detect(source:str = "videos/", weights:str = MODEL_PATH, save_txt : bool = False, img_size:int = 640, trace:bool = False,\
    project:str = "data", name:str = "live_graph", device:str = '', nosave:bool = False, augment=None, conf_thres:float = 0.25, iou_thres:float = 0.45,\
    classes=None, agnostic_nms=None, run_tracking:bool = True, unique_track_color:bool = False, view_img:bool = False, show_fps:bool = False, show_track:bool = False):

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_dir = Path(increment_path(Path(project) / name, exist_ok=True))  # increment run
    if not nosave:  
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)

    if trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, stride=stride)
    else:
        dataset = LoadImages(source, img_size=img_size, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = img_size
    old_img_b = 1

    t0 = time.time()
    ###################################
    startTime = 0
    ###################################

    videoname_to_graph_seq = dict()
    videoname_to_frame_size = dict()

    for path, img, im0s, vid_cap, videoname in dataset:
        if videoname not in videoname_to_graph_seq.keys():
            videoname_to_graph_seq[videoname] = list()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0] #TODO shouldn't it be i instead of 0?

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                dets_to_sort = np.empty((0,6))
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))


                if run_tracking:
  
                    tracked_dets = sort_tracker.update(dets_to_sort, unique_track_color)
                    tracks =sort_tracker.getTrackers()

                    # draw boxes for visualization
                    if len(tracked_dets)>0:
                        bbox_xyxy = tracked_dets[:,:4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = None

                        if show_track:
                            #loop over tracks
                            for t, track in enumerate(tracks):
                  
                                track_color = colors[int(track.detclass)] if not unique_track_color else sort_tracker.color_list[t]

                                [cv2.line(im0, (int(track.centroidarr[i][0]),
                                                int(track.centroidarr[i][1])), 
                                                (int(track.centroidarr[i+1][0]),
                                                int(track.centroidarr[i+1][1])),
                                                track_color, thickness=THICKNESS)
                                                for i,_ in  enumerate(track.centroidarr) 
                                                    if i < len(track.centroidarr)-1 ] 
                else:
                    bbox_xyxy = dets_to_sort[:,:4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]
                
                ######################################################
                is_graph = True
                graph = None
                if is_graph:
                    
                    if run_tracking:
                        bbox_xyxy = tracked_dets[:,:4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = tracked_dets[:, 5]
                    else:
                        bbox_xyxy = dets_to_sort[:,:4]
                        identities = [f"{x}_{dataset.frame}" for x in range(len(dets_to_sort))]
                        categories = dets_to_sort[:, 5]
                        confidences = dets_to_sort[:, 4]
                    
                    im0, graph = generate_spatial_graph(im0, bbox_xyxy, identities, categories, confidences, names, colors)
                ######################################################
                # im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)
                
            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            ######################################################
            if dataset.mode != 'image' and show_fps:
                currentTime = time.time()

                fps = 1/(currentTime - startTime)
                startTime = currentTime
                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)

            #######################################################
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        videoname_to_graph_seq[videoname].append(graph)
        #TODO awful temporary solution to transmit height and width
        videoname_to_frame_size[videoname] = (w,h)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
    print(f'Done. ({time.time() - t0:.3f}s)')

    return videoname_to_graph_seq, videoname_to_frame_size
    

