'''
UTILITY FUNCTIONS
This file contains utility functions used in model pre-training and evaluation.
'''

# import numpy
import numpy as np
import pandas as pd

# scikit-learn metrics
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score


# CALCULATE METRICS
def calculate_metrics(pred, target, threshold = 0.5):
    '''
    Calculates accuracy, average precision, F1 score, and ROC AUC score.

    Args:
        pred (numpy.ndarray): Array of predicted values.
        target (numpy.ndarray): Array of target values.
        threshold (float): Threshold for prediction.
    '''

    # compute metrics
    accuracy = accuracy_score(target, pred > threshold) # accuracy
    ap = average_precision_score(target, pred) # average precision score
    f1 = f1_score(target, pred > threshold, average = 'micro') # F1 score
    try: 
        auroc = roc_auc_score(target, pred) # ROC AUC score
    except ValueError: 
        auroc = 0.5 

    # create and return dictionary of metrics
    metrics = {'accuracy': accuracy, 'ap': ap, 'f1': f1, 'auroc': auroc}
    return metrics


# RANDOM WALK
def random_walk(edges, seed_node, walk_length, walk_nodes = None):
    '''
    Performs a random walk on the input graph.

    Args:
        edges (pandas.DataFrame): Data frame of edges in graph.
        seed_node (int): Index of seed node.
        walk_length (int): Length of random walk.
        walk_nodes (set): Set of nodes in random walk.
    '''

    # if walk length is 0, return nodes
    if walk_length == 0:
        return walk_nodes

    # initialize list to store random walk
    if walk_nodes is None:
        walk_nodes = {seed_node}
    else:
        walk_nodes.add(seed_node)

    # get neighbors of seed node
    neighbors = edges[edges.x_index == seed_node].y_index.values
    neighbor = np.random.choice(neighbors)

    # return random walk
    return random_walk(edges, neighbor, walk_length - 1, walk_nodes)


# RANDOM SUBGRAPH
def random_subgraph(edges, seed_node, n_walks, walk_length):
    '''
    Generates a random subgraph from the input graph.

    Args:
        edges (pandas.DataFrame): Data frame of edges in graph.
        seed_node (int): Index of seed node.
        n_walks (int): Number of random walks to perform.
        walk_length (int): Length of each random walk.
    '''

    # generate set of subgraph nodes
    subgraph_nodes = {seed_node}

    # iterate over number of walks
    for i in range(n_walks):

        # pick random node in subgraph as seed
        seed_node_i = np.random.choice(list(subgraph_nodes))

        # perform random walk
        subgraph_nodes = random_walk(edges, seed_node_i, walk_length, subgraph_nodes)

        # add nodes to subgraph
        subgraph_nodes.update(subgraph_nodes)

    # return subgraph nodes
    return subgraph_nodes


# function to generate random subgraph
def generate_subgraph(nodes, edges, seed_node, n_walks, walk_length):
    '''
    Generates a random subgraph from the input graph.
    Then, re-map the node indices to be consecutive zero-indexed integers.

    Args:
        nodes (pandas.DataFrame): Data frame of nodes in graph.
        edges (pandas.DataFrame): Data frame of edges in graph.
        seed_node (int): Index of seed node.
        n_walks (int): Number of random walks to perform.
        walk_length (int): Length of each random walk.
    '''

    # generate random subgraph
    subgraph = random_subgraph(edges, 1, 100, 10)

    # subset data frame for subgraph nodes and edges
    subgraph_nodes = nodes[nodes.node_index.isin(subgraph)].copy()
    subgraph_edges = edges[edges.x_index.isin(subgraph) & edges.y_index.isin(subgraph)].copy()

    # assign new node indices, store old node indices in original_node_index
    subgraph_nodes['original_node_index'] = subgraph_nodes['node_index']
    subgraph_nodes['node_index'] = range(len(subgraph_nodes))

    # assign new edge indices, store old edge indices in original_x_index and original_y_index
    subgraph_edges['original_x_index'] = subgraph_edges['x_index']
    subgraph_edges['original_y_index'] = subgraph_edges['y_index']
    subgraph_edges['x_index'] = subgraph_edges['x_index'].map(subgraph_nodes.set_index('original_node_index')['node_index'])
    subgraph_edges['y_index'] = subgraph_edges['y_index'].map(subgraph_nodes.set_index('original_node_index')['node_index'])

    # reindex nodes and edges
    subgraph_nodes = subgraph_nodes.reset_index(drop = True)
    subgraph_edges = subgraph_edges.reset_index(drop = True)

    # return nodes and edges
    return subgraph_nodes, subgraph_edges