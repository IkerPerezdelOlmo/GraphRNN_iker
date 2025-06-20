import networkx as nx
import numpy as np

from utils import *
from data import *


def create(args):
    return create_datasetName(args.graph_type, args)


def create_datasetName(name, args):

    graphs=[]
    # synthetic graphs
    if name == 'random50':
        graphs = Graph_load_batch(min_num_nodes = 2, max_num_nodes = 100, name = 'random50',node_attributes = False ,graph_labelsUsed=True)
        args.max_prev_node = 89   # porque vamos a tener todos los anteriores nodos en cuenta

    elif name == 'random100':
        graphs = Graph_load_batch(min_num_nodes = 2, max_num_nodes = 100, name = 'random100',node_attributes = False ,graph_labelsUsed=True)
        name = 89   # porque vamos a tener todos los anteriores nodos en cuenta

    elif name == 'random500':
        graphs = Graph_load_batch(min_num_nodes = 2, max_num_nodes = 100, name = 'random500',node_attributes = False ,graph_labelsUsed=True)
        args.max_prev_node = 89   # porque vamos a tener todos los anteriores nodos en cuenta

    elif name == 'random1000':
        graphs = Graph_load_batch(min_num_nodes = 2, max_num_nodes = 100, name = 'random1000',node_attributes = False ,graph_labelsUsed=True)
        args.max_prev_node = 89   # porque vamos a tener todos los anteriores nodos en cuenta
    else:
        izena = name.split("_")
        graphs = Graph_load_batch(min_num_nodes = 2, max_num_nodes = int(izena[3][1:])+1, name = name ,node_attributes = False ,graph_labelsUsed=True)
        args.max_prev_node = int(izena[3][1:])+1   # porque vamos a tener todos los anteriores nodos en cuenta

    # Include other graph types as per your existing implementation
    return graphs

