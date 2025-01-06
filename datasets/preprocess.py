import dgl
import pandas as pd
import numpy as np
import os
import sys
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import dgl
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import ast
import os
import dgl
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from dgl.data.utils import save_graphs
from torch_geometric import data as DATA
from torch_geometric.data import Batch
import json
import networkx as nx
from torch_geometric.utils import subgraph
from randomWalk.node2vec import Node2vec


class HeteroKG(object):
    def __init__(self, kg_path, graph_path):
        self.kg_path = os.path.join(kg_path, 'kg.csv')
        self.graph_path = graph_path
        self.node_info = os.path.join(kg_path, 'node_info.csv')
        #self.logger = logger
    
    def read_kg(self):
        if os.path.exists(self.graph_path):
            glist, _ = dgl.load_graphs(self.graph_path)
            graph = glist[0]
            unique_nodes_df = pd.read_csv(self.node_info, index_col=0)
        else:
            with open(self.kg_path, 'r') as f:
                kg_df = pd.read_csv(f, low_memory=False)
            graph, unique_nodes_df = self.create_heterogeneous_graph_by_primekg(kg_df)
            save_graphs(self.graph_path, [graph], {"glabel": torch.tensor([0])})
            unique_nodes_df.to_csv(self.node_info)
        print("Knowledge Graph has {} nodes with {} edges among {} relationships".format(graph.num_nodes(), graph.num_edges(), len(set(graph.etypes))))
        #self.logger.info("Knowledge Graph has {} nodes with {} edges among {} relationships".format(graph.num_nodes, graph.num_edges, len(set(graph.etypes))))

        return graph, unique_nodes_df

    def create_heterogeneous_graph_by_primekg(self, kg_df):
        """
        Create the data sources for the HGT model - which are the hetrogenous knoweldge graph and the unique nodes DataFrame, with the mapping between the original node indices and the graph ones (by node type).
        kg_df: df. the dataframe of the knowlege graph
        """
        # creating the nodes DataFrame
        unique_nodes_df = self.create_nodes_df(kg_df)

        # creating the mapping between the original node indices and the graph ones (by node type)
        unique_nodes_df['node_type_graph_index'] = unique_nodes_df.groupby('node_type').cumcount()

        # creating the hetrogenous graph
        kg = self.create_hetro_graph(kg_df, unique_nodes_df)

        # setting the graph's node features which are the original indices. 
        grouped_nodes = unique_nodes_df.groupby('node_type', sort = False)

        # create the features (which are the original nodes indices)
        # iterate over the groups and add global node indices to the graph
        for node_type, nodes_subset in grouped_nodes:
            # nodes_subset_new = nodes_subset.iloc[:len_subgraph] # find this row to be weird
            kg.nodes[node_type].data['node_index'] = torch.tensor(nodes_subset['node_index'].values)

        return kg, unique_nodes_df

    def create_nodes_df(self, kg_df):
        """
        Create a DataFrame of unique nodes in the knowledge graph.
        kg_df: df. the dataframe of the knowlege graph
        """
        node_x_df = kg_df[['x_index', 'x_id', 'x_type', 'x_name']].rename(
            columns={'x_index': 'node_index', 'x_id': 'node_id', 'x_type': 'node_type', 'x_name': 'node_name'}
        )
        
        node_y_df = kg_df[['y_index', 'y_id', 'y_type', 'y_name']].rename(
            columns={'y_index': 'node_index', 'y_id': 'node_id', 'y_type': 'node_type', 'y_name': 'node_name'}
        )
        
        # Combine both DataFrames
        all_nodes_df = pd.concat([node_x_df, node_y_df], ignore_index=True)
        
        # Drop duplicates to get unique nodes
        unique_nodes_df = all_nodes_df.drop_duplicates()
        
        # Reset index (optional)
        unique_nodes_df.reset_index(drop=True, inplace=True)

        return unique_nodes_df

    def create_hetro_graph(self, kg_df, nodes_df):
        """
        Create a DGL HeteroGraph from the knowledge graph DataFrame and the unique nodes DataFrame.
        kg_df: df. the dataframe of the knowlege graph
        nodes_df: df. the dataframe of the unique nodes in the knowledge graph
        """
        # Use the 'node_type_index' column to create the 'x_type_index' and 'y_type_index' columns in the edges DataFrame
        x_node_type_graph_index = pd.merge(kg_df, nodes_df, left_on='x_index', right_on='node_index', how='left')[['node_type_graph_index']].rename(
            columns={'node_type_graph_index': 'x_node_type_graph_index'}
        )
        y_node_type_graph_index = pd.merge(kg_df, nodes_df, left_on='y_index', right_on='node_index', how='left')[['node_type_graph_index']].rename(
            columns={'node_type_graph_index': 'y_node_type_graph_index'}
        )

        ##print(x_node_type_graph_index)
        
        kg_df['x_node_type_graph_index'] = x_node_type_graph_index
        kg_df['x_node_type_graph_index'] = kg_df['x_node_type_graph_index'].astype(int)
        kg_df['y_node_type_graph_index'] = y_node_type_graph_index
        kg_df['y_node_type_graph_index'] = kg_df['y_node_type_graph_index'].astype(int)

        #print(kg_df.head(10))

        # Define empty dictionary to store graph data
        kg_data = {}

        # Group the edges DataFrame by unique combinations of x_type, relation, and y_type
        grouped_edges = kg_df.groupby(['x_type', 'relation', 'y_type'], sort = False)

        # Iterate over the groups
        for (x_type, relation, y_type), edges_subset in grouped_edges:
            # Convert edge indices to torch tensor
            edge_indices = (torch.tensor(edges_subset['x_node_type_graph_index'].values), torch.tensor(edges_subset['y_node_type_graph_index'].values))

            # Add edge indices to data object
            kg_data[(x_type, relation, y_type)] = edge_indices

            # Print update
            # print(f'Added edge relation: {x_type} - {relation} - {y_type}')

        # Instantiate a DGL HeteroGraph
        kg = dgl.heterograph(kg_data)

        return kg
    
    def get_type_graph_index(self, org_idx, unique_nodes_df):
        """
        Get the graph index of a node given its original index.
        org_idx: int. the original index of the node
        unique_nodes_df: df. the dataframe of the unique nodes in the knowledge graph
        """
        return unique_nodes_df.loc[unique_nodes_df['node_index'] == org_idx]['node_type_graph_index'].values[0]


class LoadData(object):
    def __init__(self, pre_path, subgraph_generator, kg, max_nodes = 500, khop=2, path_len=2):
        self.pre_path = pre_path
        self.subgraph_generator = subgraph_generator
        self.kg = kg  ##DGL object
        self.max_nodes = max_nodes
        self.path_len = path_len
        self.khop=khop

        self.dict_path = os.path.join(pre_path, 'dictionary.json')
        self.mapped_path = os.path.join(pre_path, 'mapped_code.csv')
        self.dictionary, self.mapped_code = self.read_data()
    
    def read_data(self):
        with open(self.dict_path, 'r') as f:
            dictionary = json.load(f)
        
        mapped_code_dict = {}
        mapped_code = pd.read_csv(self.mapped_path)
        cuis, index = mapped_code['CUI'].values, mapped_code['index'].values
        for cui, idx in zip(cuis, index):
            mapped_code_dict[str(cui)] = idx
        return dictionary, mapped_code
    
    def get_description(self, code):
        ##using LLM to generate the description
        return;

    def get_mapped_code(self, code_cui):
        ##using the dictionary to get the mapped code
        return self.dictionary[code_cui]
    
    def get_kg(self):
        rel_types = self.kg.etypes
        print("Relation types:", rel_types)
        # 为每种关系分配一个 ID
        rel_mapping = {rel: i for i, rel in enumerate(rel_types)}
        print("Relation mapping:", rel_mapping)

        rel_index = []
        for rel in rel_types:
            src, dst = self.kg.edges(etype=rel)
            edge_index = torch.stack([src, dst], dim=0)
            rel_index.extend(rel_mapping[rel] * len(src))
            print(f"Edge index for relation '{rel}':\n", edge_index)
        
        rel_index = torch.tensor(rel_index)

        return edge_index, rel_index

    
    def get_code_subgraph(self, target_node):
        edge_index, rel_index = self.get_kg()
        if self.subgraph_generator == 'rw':
            return self.get_subgraph(target_node, edge_index, rel_index)
        elif self.subgraph_generator == 'k_hop':
            return self.get_k_hop_subgraph(target_node)
        elif self.subgraph_generator == 'pagerank':
            return self.get_pagerank_subgraph(target_node)

    def get_subgraph_rw(self, target_node, edge_index, rel_index):

        json_path = "rw_num_3_length_" + str(self.path_len) + "sp.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                subgraphs = json.load(f)
            return subgraphs
        
        my_graph = nx.Graph()
        my_graph.add_edges_from(edge_index.transpose(1,0).numpy().tolist())
        undirected_rel_index = torch.cat((rel_index, rel_index), 0)

        subgraphs = {}
        for d in target_node:
            subsets = Node2vec(start_nodes=[int(d)], graph=my_graph, path_length=self.path_len, num_paths=3, workers=6, dw=True).get_walks() ##返回一个list
            print(subsets)
            mapping_id = subsets.index(int(d))
            mapping_list = [False for _ in range(len((subsets)))]
            mapping_list[mapping_id] = True

            sub_edge_index, sub_rel_index = subgraph(subsets, edge_index, undirected_rel_index, relabel_nodes=True)
            
            new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
            new_s_rel = sub_rel_index.numpy().tolist()
            subgraphs[d] = subsets, new_s_edge_index, new_s_rel, mapping_list

        with open(json_path, 'w') as f:
            json.dump(subgraphs, f, default=convert)

        return subgraphs
    
    def get_pagerank_subgraph(self, target_node, edge_index, rel_index):

        json_path = "prob_fix_" + str(self.max_nodes) + "_sp.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                subgraphs = json.load(f)
            return subgraphs;

        g = nx.DiGraph()
        g.add_edges_from(edge_index.transpose(1, 0).tolist())

        pagerank_path = "pagerank.json"
        if not os.path.exists(pagerank_path):
            pagerank = np.array(google_matrix(g), dtype='float32')
            page_dict = {}
            for d in target_node:
                page_dict[d] = list(pagerank[list(g.nodes()).index(int(d))])
            with open(pagerank_path, 'w') as f:
                json.dump(page_dict, f)
        else:
            with open(pagerank_path, 'r') as f:
                page_dict = json.load(f)
            f.close()

        undirected_rel_index = torch.cat((rel_index, rel_index), 0)

        subgraphs = {}
        for d in target_node:
            subsets = [int(d)]

            neighbors = np.random.choice(
                a=list(g.nodes()),
                size=self.max_nodes,
                replace=False,
                p=page_dict[d])

            subsets.extend(neighbors)
            subsets = list(set(subsets))

            mapping_list = [False for _ in subsets]
            mapping_idx = subsets.index(int(d))
            mapping_list[mapping_idx] = True

            sub_edge_index, sub_rel_index =  subgraph(subsets, edge_index, undirected_rel_index, relabel_nodes=True)
            new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
            new_s_rel = sub_rel_index.numpy().tolist()

            subgraphs[d] = subsets, new_s_edge_index, new_s_rel, mapping_list

        with open(json_path, 'w') as f:
            json.dump(subgraphs, f, default=convert)

        return subgraphs

def convert(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

def google_matrix(
    G, alpha=0.85, personalization=None, nodelist=None, weight="weight", dangling=None
):
    import numpy as np

    if nodelist is None:
        nodelist = list(G)

    M = np.asmatrix(nx.to_numpy_array(G, nodelist=nodelist, weight=weight), dtype='float32')
    N = len(G)
    if N == 0:
        return M

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N).astype('float32')
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype="float32")
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()


    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype='float32')
        dangling_weights /= dangling_weights.sum()
    dangling_nodes = np.where(M.sum(axis=1) == 0)[0]

    # Assign dangling_weights to any dangling nodes (nodes with no out links)
    for node in dangling_nodes:
        M[node] = dangling_weights

    M /= M.sum(axis=1).astype('float32')  # Normalize rows to sum to 1

    return np.multiply(alpha, M, dtype='float32') + np.multiply(1 - alpha, p, dtype='float32')

class MedCodeDataset(Dataset):
    def __init__(self, kg_path, graph_save_path, med_codes_pkg_map_path):
        self.kg_path = kg_path
        self.med_codes_pkg_map_path = med_codes_pkg_map_path
        hetro_kg = HeteroKG(kg_path, graph_path=graph_save_path)
        graph, unique_nodes_df = hetro_kg.read_kg()
        self.g = graph
        self.unique_nodes_df = unique_nodes_df
        self.med_codes_pkg_df = pd.read_parquet(med_codes_pkg_map_path)
    
    def get_data(self, idx):
        nodes_l = self.med_codes_pkg_df.iloc[idx]['pkg_index_list']
        med_code = self.med_codes_pkg_df.iloc[idx]['med_code']
        subgraph = self.get_subgraph_hetro(nodes_l)

        ##here to get med_code tokens and add the padding, we first replace it with the nodes_l

        data = DATA.Data(x=torch.LongTensor(idx),  ##medcode idx
                         text = torch.LongTensor(nodes_l),  ##medcode tokens
                         dgl_graph = subgraph,
                        )
        
        return data
    
    def __len__(self):
        return len(self.med_codes_pkg_df)

    def __getitem__(self, idx):
        return self.get_data(idx)

    def get_subgraph_hetro(self, nodes_l):
        nodes_group_df = self.unique_nodes_df[self.unique_nodes_df['node_index'].isin(nodes_l)]
        # print("nodes_group_df: ", nodes_group_df.shape)
        selected_nodes_groups = nodes_group_df.groupby('node_type')
        
        filter_dict = {}
        # creating the dict per node type for the subgraph creation
        for node_type, split_subset in selected_nodes_groups:
            # print(node_type)
            filter_dict[node_type] = split_subset['node_type_graph_index'].tolist()
    
        sg = dgl.node_subgraph(self.g, filter_dict) # relabel_nodes=False will keep all the nodes from the original kg.
        return sg
        

def custom_collate_fn(data_list):
    batch_data = Batch.from_data_list(data_list)
    return batch_data


if __name__ == "__main__":
    #kg_path = '/n/data1/hms/dbmi/zitnik/lab/users/shvat372/ICML_codes/kg.csv'
    #med_codes_pkg_map_path = '/n/data1/hms/dbmi/zitnik/lab/users/shvat372/ICML_codes/graphs/toy_mappings.parquet'
    #graph_save_path = '/n/data1/hms/dbmi/zitnik/lab/users/shvat372/ICML_codes/kg_temp_2912'

    kg_path = '/n/netscratch/mzitnik_lab/Lab/xsu/primeKG/'
    med_codes_pkg_map_path = '/n/netscratch/mzitnik_lab/Lab/xsu/toy_mappings.parquet'
    graph_save_path = '/n/netscratch/mzitnik_lab/Lab/xsu/kg_temp_2912'

    medcode_dataset = MedCodeDataset(kg_path, graph_save_path, med_codes_pkg_map_path)
    dataloader = DataLoader(medcode_dataset, batch_size=1, num_workers=0, collate_fn=custom_collate_fn)

    for med_codes, sgs in tqdm(dataloader):
        print(med_codes)
        print(sgs)
        break







