from st_graph_generation import detect_with_networkx
import torch
import time
import argparse
from anomaly_generation.graph_corruption import Corruptor


COMMAND = "python3 src/pipeline.py --no-trace --view-img --source test/street.mp4 --show-fps --track --show-track --project data --name live_graph"

'''
Use the following only to document the parameters of the detect function or to understand what you can remove etc.
def prepare_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default=detect_with_networkx.MODEL_PATH, help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='test/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    #parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracked path')
    parser.add_argument('--show-fps', action='store_true', help='show fps')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')


    opt = parser.parse_args()
    print(opt)
    return opt
'''

''''
HOW CORRUPTUION CODE SHOULD BE CALLED
for each video clip:
    corruptor = Corruptor(..., ...)
    normal_sequence = list()
    corrupted_sequence = list()

    for frame in clip:
        graph = //graph initialization
        corrupted_graph = corruptor.corrupt_graph(graph)
        normal_sequence.append(graph)
        corrupted_sequence.append(corrupted_graph)
        // do we need object tracking etc.???
'''

if __name__=='__main__':
    with torch.no_grad():
        #t = time.time()
        videoname_to_graph_seq, videoname_to_frame_size = detect_with_networkx.detect()
        #print(f'Done. ({time.time() - t:.3f}s)')

    
    for videoname in videoname_to_graph_seq:
        print(len(videoname_to_graph_seq[videoname]))



    videoname_to_corr_graph_seq = dict()
    for videoname in videoname_to_graph_seq:
        w, h = videoname_to_frame_size[videoname]
        #TODO so far, is_stg must be true because only the distance is implemented on the edge as 'weight'
        corruptor = Corruptor(frame_width=w, frame_height=h, is_stg=True)

        videoname_to_corr_graph_seq[videoname] = list()
        
        for graph in videoname_to_graph_seq[videoname]:
            corrupted_graph = corruptor.corrupt_graph(graph)
            videoname_to_corr_graph_seq[videoname].append(corrupted_graph)
        
    videoname_to_corr_graph_seq