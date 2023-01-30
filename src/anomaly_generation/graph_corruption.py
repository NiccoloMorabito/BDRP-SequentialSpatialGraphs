import networkx as nx
import random
from copy import deepcopy
from common_utils.stg_utils import *

class Corruptor:
    def __init__(self, frame_height: int, frame_width: int, corruption_categs: bool = True, \
        corruption_bboxes : bool = True, corruption_weights: bool = True, permutation_weights : bool = False):
        self.frame_height = frame_height
        self.frame_width = frame_width
        #TODO these booleans (and their parameters?) could also be generated randomly (but use the same for the same Corruptor)
        self.corruption_categs, self.corruption_bboxes, self.corruption_weights, self.permutation_weights = \
            corruption_categs, corruption_bboxes, corruption_weights, permutation_weights

        # memory fields (when a node is not in the graph anymore, it must be removed from here (its id could be reassigned by yolo))
        self.nodeid_to_category = dict()
        self.nodeid_to_bbox_params = dict()
    
    def corrupt_graph(self, graph: nx.Graph, verbose: bool = False):
        copy = deepcopy(graph)
        
        if self.corruption_categs:
            if verbose: print(f"Corrupting categories...\n\tfrom:{[n.class_name for n in copy.nodes]}")
            self.__corrupt_categories(copy)
            if verbose: print(f"\tto: {[n.class_name for n in copy.nodes]}")

        if self.corruption_weights:
            if verbose: print(f"Corrupting weights...\n\tfrom:{[d for _, _, d in copy.edges(data=True)]}")
            self.__corrupt_weights(copy)
            if verbose: print(f"\tto:{[d for _, _, d in copy.edges(data=True)]}")

        if self.permutation_weights:
            if verbose: print(f"Permutating weights...\n\tfrom:{[(n1.id, n2.id, d) for n1, n2, d in copy.edges(data=True)]}")
            self.__permute_weights(copy)
            if verbose: print(f"\tto:{[(n1.id, n2.id, d) for n1, n2, d in copy.edges(data=True)]}")
        
        if self.corruption_bboxes:
            if verbose: print(f"Corrupting boundary boxes...\n\tfrom:{[(n.x1, n.y1, n.x2, n.y2) for n in copy.nodes]}")
            self.__corrupt_boundary_boxes(copy)
            if verbose: print(f"\tto:{[(n.x1, n.y1, n.x2, n.y2) for n in copy.nodes]}")

        return copy
    
    def __corrupt_categories(self, graph: nx.Graph, corruption_prob: float = 0.4):
        '''
        Corrupt categories (i.e. for each frame/graph, if the object is new decide
            - whether to corrupt its category from that moment on
            - or never do
        '''
        #TODO (optionally) if the object is not new, you can also revaluate with a smaller
        # probability whether to corrupt its category from that moment on
        for node in graph.nodes:
            # if the object is new randomly choose whether to corrupt the category or not
            if node.id not in self.nodeid_to_category.keys():
                if random.random() < corruption_prob:
                    self.nodeid_to_category[node.id] = random.choice(CATEGORIES)
                else:
                    self.nodeid_to_category[node.id] = node.class_name
            node.class_name = self.nodeid_to_category[node.id]
        
        # remove the ids of the objects that are no longer in the frame
        present_ids = [n.id for n in graph.nodes]
        self.nodeid_to_category = {k: v for k, v in self.nodeid_to_category.items() if k in present_ids}
    
    def __corrupt_weights(self, graph: nx.Graph, k: int = None):
        '''
        Corrupt weights of the graph (i.e. for every frame, reduce the distance between objects wrt to the previous frame)
        '''
        # choose the set of edges to corrupt the weights of
        edges_to_corrupt = list(graph.edges(data=True))
        if k is not None:
            edges_to_corrupt = random.sample(edges_to_corrupt, k=k)

        # TODO clean this code
        for n1, n2, d in edges_to_corrupt:
            reducing_percentage = random.random() #TODO this could become a paramter to control the reduction
            # reduce the weight of the edge since it represents the distance
            d[EDGE_LABEL] -= reducing_percentage * d[EDGE_LABEL]

    def __permute_weights(self, graph: nx.Graph):
        # this function is meaningless in case the graph is fully connected
        # since only the weights are permuted, __corrupt_weights() is enough
        if self.is_stg:
            weights = [d[EDGE_LABEL] for _, _, d in graph.edges(data=True)]
            random.shuffle(weights)

            for index, edge in enumerate(graph.edges(data=True)):
                n1, n2, d = edge
                d[EDGE_LABEL] = weights[index]
        else:
            distances = [d[EDGE_LABEL] for _, _, d in graph.edges(data=True)]
            speeds = [d[EDGE_LABEL] for _, _, d in graph.edges(data=True)]
            # shuffle the two lists keeping the relationships
            z = list(zip(distances, speeds))
            random.shuffle(z)
            distances, speeds = zip(*z)

            for index, edge in enumerate(graph.edges(data=True)):
                n1, n2, d = edge
                d[EDGE_LABEL] = distances[index]
                d[EDGE_LABEL] = speeds[index]
    
    def __corrupt_boundary_boxes(self, graph: nx.Graph, corruption_prob: float = 0.4):
        '''
        Corrupt boundary boxes (i.e. randomly change the dimension of some randomly chosen objects in random frames.
        '''
        # this method will consequently affect also the area feature
        #TODO it is also possible to just make the objects bigger (smaller it's probably no anomaly)
        for node in graph.nodes:
            # if the object is new randomly choose whether to corrupt the boundary box or not
            if node.id not in self.nodeid_to_bbox_params.keys():
                if random.random() < corruption_prob:
                    self.nodeid_to_bbox_params[node.id] = self.__generate_random_bbox_params()
                else:
                    continue #this means that an object could be corrupted later
            
            bigger, perc_x1, perc_x1, perc_y1, perc_x2, perc_y2 = self.nodeid_to_bbox_params[node.id]

            '''
            #TODO the testing code can work only after fixing the bbox originally
            # it seems that x2 or y2 are at most 5 out of the width/height
            # the following code prints the problems:
            if node.x1 > node.x2 or node.x2 > self.frame_width:
                print(node.x1, node.x2, self.frame_width)
            if node.y1 > node.y2 or node.y2 > self.frame_height:
                print(node.y1, node.y2, self.frame_height)
            ### Testing ###
            bbox = (node.x1, node.y1, node.x2, node.y2)
            assert node.x1 < node.x2 <= self.frame_width, f"Error in original bbox (x): {node.x1, node.x2, self.frame_width}"
            assert node.y1 < node.y2 <= self.frame_height, f"Error in original bbox (y): {node.y1, node.y2, self.frame_height}"
            ### Testing ###
            '''

            
            if bigger:
                node.x1 -= int(perc_x1 * node.x1) 
                node.y1 -= int(perc_y1 * node.y1)
                node.x2 += int(perc_x2 * (self.frame_width-node.x2)) 
                node.y2 += int(perc_y2 * (self.frame_height-node.y2))
            else:
                node.x1 += int(perc_x1 * (node.x2-node.x1))
                node.y1 += int(perc_y1 * (node.y2-node.y1))
                node.x2 -= int(perc_x2 * (node.x2-node.x1))
                node.y2 -= int(perc_y2 * (node.y2-node.y1))
            
            '''
            #TODO code to check the testing that fail:
            if node.x1 > node.x2 or node.x2 > self.frame_width:
                print(node.x1, node.x2, self.frame_width)
            if node.y1 > node.y2 or node.y2 > self.frame_height:
                print(node.y1, node.y2, self.frame_height)
            ### Testing ###
            bbox = (node.x1, node.y1, node.x2, node.y2)
            assert min(bbox) >= 0, "Error in corrupted bbox (negative)"
            assert node.x1 < node.x2 <= self.frame_width, f"Error in corrupted bbox (x): {node.x1, node.x2, self.frame_width}"
            assert node.y1 < node.y2 <= self.frame_height, f"Error in corrupted bbox (y): {node.y1, node.y2, self.frame_height}"
            ### Testing ###
            '''
    
            node.centroid = ((node.x1 + node.x2) // 2, (node.y1 + node.y2) // 2)
        
        # remove the ids of the objects that are no longer in the frame
        present_ids = [n.id for n in graph.nodes]
        self.nodeid_to_bbox_params = {k: v for k, v in self.nodeid_to_bbox_params.items() if k in present_ids}

    def __generate_random_bbox_params(self):
        #TODO this generation could be changed to make the results more "controlled" (for instance, reducing the interval of percentages to [0,0.5])
        bigger = bool(random.getrandbits(1))
        perc_x1, perc_y1, perc_x2, perc_y2 = random.random(), random.random(), random.random(), random.random()
        return bigger, perc_x1, perc_x1, perc_y1, perc_x2, perc_y2            

