'''
POSSIBLE IDEAS FOR CORRUPTION (to develop, update, etc.)

- [only if the graph is NOT fully connected] permutate edges (like Anograph) [only if the graph is not fully connected]
- [only if the graph is NOT fully connected] generating or removing edges

- change weight of the edges
    - for every frame, increase the speed/reduce the distance wrt to the previous frame
- randomly corrupt categories
    (- in a sequence, from the first frame where there are objects randomly choose some and corrupt their categories from that moment on
    (- in a sequence, from a random frame where there are objects, randomly choose some and corrupt their categories from that moment on
    (- during the sequence, whenever a new object comes out decide randomly whether to corrupt its category from that moment on
    summary of all these options: for each frame, if the object is new decide whether to corrupt its category from that moment on
        (optionally) if the object is not new, you can also revaluate with a smaller probability whether to corrupt its cat from that moment on #TODO
- change boundary boxes
    - randomly change (increase or decrease) dimension of some randomly chosen objects in random frames

- add nodes #TODO (the problem is how to track the added objects with object tracking ->
        if nodes are added, provide the list of added nodes (how to know the object tracker id from generation code?)
    - at a random time, generate a still object for the following frames #TODO
    - at a random time, generate an object and make it move every frame #TODO
'''

import networkx as nx
import random
from copy import deepcopy
from common_utils.stg_utils import *

class Corruptor:
    def __init__(self, frame_height: int, frame_width: int, is_stg: bool = True) -> None:
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.is_stg = is_stg # TODO if stg, the edge has only one attribute (distance). otherwise, it has two (distance and speed)

        #TODO randomly choose these booleans (and their parameters?) but remember to use the same boolean for the same Corruptor
        self.corruption_categs, self.corruption_weights, self.permutation_weights, self.corruption_bboxes = True, True, False, True

        # memory fields (TODO remember that when a node is not in the graph anymore, it must be removed from here (its id could be reassigned))
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
    
    # this function is probably unaffected by the temporal sequence (a frame can have completely different weights than before)
    def __corrupt_weights(self, graph: nx.Graph, k: int = None):
        # choose the set of edges to corrupt the weights of
        edges_to_corrupt = list(graph.edges(data=True))
        if k is not None:
            edges_to_corrupt = random.sample(edges_to_corrupt, k=k)

        # TODO clean this code
        for n1, n2, d in edges_to_corrupt:
            reducing_percentage = random.random()
            # spatial edge has only one weight (distance)
            if self.is_stg:
                d[STG_SPATIAL_DIST_LABEL] -= reducing_percentage * d[STG_SPATIAL_DIST_LABEL]
            # spatial edge has two weights (distance and speed)
            else:
                d[SG_SPATIAL_DIST_LABEL] -= reducing_percentage * d[SG_SPATIAL_DIST_LABEL]
                d[SG_SPATIAL_SPEED_LABEL] += reducing_percentage * d[SG_SPATIAL_SPEED_LABEL]
    
    # this function is probably unaffected by the temporal sequence (a frame can have completely different weights than before)
    # this function could be meaningless if the graph is fully connected (since only the weights are permuted)
    def __permute_weights(self, graph: nx.Graph):
        if self.is_stg:
            weights = [d[STG_SPATIAL_DIST_LABEL] for _, _, d in graph.edges(data=True)]
            random.shuffle(weights)

            for index, edge in enumerate(graph.edges(data=True)):
                n1, n2, d = edge
                d[STG_SPATIAL_DIST_LABEL] = weights[index]
        else:
            distances = [d[SG_SPATIAL_DIST_LABEL] for _, _, d in graph.edges(data=True)]
            speeds = [d[SG_SPATIAL_SPEED_LABEL] for _, _, d in graph.edges(data=True)]
            # shuffle the two lists keeping the relationships
            z = list(zip(distances, speeds))
            random.shuffle(z)
            distances, speeds = zip(*z)

            for index, edge in enumerate(graph.edges(data=True)):
                n1, n2, d = edge
                d[SG_SPATIAL_DIST_LABEL] = distances[index]
                d[SG_SPATIAL_SPEED_LABEL] = speeds[index]
    
    def __corrupt_boundary_boxes(self, graph: nx.Graph, corruption_prob: float = 0.4):
        #TODO this method is only effective if the bboxes are embedded in the graph
        #TODO this method is probably more effective if the area resulting from the bboxes is also embedded in the graph
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




'''
#TODO remove this function as soon as you have the real code
def dummy_graph():
    graph = nx.Graph()
    add_random_nodes_in(graph, 5)
    
    for n1 in graph.nodes:
        for n2 in graph.nodes:
            if n1!=n2:
                graph.add_edge(n1, n2, weight=distance.euclidean(n1.centroid, n2.centroid))
                #print(graph.edges[n1, n2]['weight'])

    return graph
'''

'''
CODE FOR CORRUPTING THROUGH ADDITION OF NODES (prob I won't implement it)

#TODO make it unique?
def random_id():
    return int(str(uuid.uuid4().fields[-1]))

#TODO see if you can implement this or not
# for making this function consistent in time, the set of generated nodes should be kept for the following frames
# however, this would make the objects still
def add_random_nodes_in(graph: nx.Graph, k: int) -> nx.Graph: #TODO missing arguments: img_width: int, img_height: int
    # TODO decide how to connect these nodes
    for _ in range(k):
        id = random_id()
        #TODO according to the meaning of x1,x2,y1,y2 you need to multiply the random percentage to the width/height of the image
        # to get the first point, and then multiply the percentage only to the difference (the available height/width) for the other
        x1=int(random.random()*100)
        y1=int(random.random()*100)
        x2=x1*2
        y2=int(y1*1.5)
        category = random.choice(CATEGORIES)
        graph.add_node(Node(id, x1, y1, x2, y2, conf=0, detclass="", class_name=category, centroid=((x1 + x2) // 2, (y1 + y2) // 2)))
'''

                


'''
if __name__=='__main__':
    
    g = dummy_graph()
    g_copy = deepcopy(g)
    g_copy.remove_node(list(g.nodes)[-1])
    l = [g, g_copy]
    
    
    corruptor = Corruptor(500, 500, True)
    for g in l:
        h = corruptor.corrupt_graph(g)
'''