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
from torch_geometric.utils import subgraph
from transformers import AutoTokenizer, pipeline

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

class EdgeDropout:
    def __init__(self, p: float = 0.1):
        """
        Drop a fraction of edges randomly.
        Args:
            p (float): Probability of dropping each edge. Default: 0.1
        """
        self.p = p

    def __call__(self, edge_index, rel_index):
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) > self.p
        edge_index_aug = edge_index[:, mask]
        rel_index_aug = rel_index[mask]

        return edge_index_aug, rel_index_aug



class MedCodeDataset(Dataset):
    def __init__(self, kg_path, graph_path, med_codes_pkg_map_path, text_model, max_length=512):
        self.kg_path = kg_path
        self.med_codes_pkg_map_path = med_codes_pkg_map_path
        
        self.med_codes_pkg_df = pd.read_parquet(med_codes_pkg_map_path)
        self.text_model = text_model
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model)
        self.max_length = max_length
        self.edge_index, self.rel_index = self.get_kg()
        self.transform = EdgeDropout(p=0.1)

        #self.med_code_graph = self.get_med_graph()
        #print("med_code_graph: ")
        self.med_code_des = self.get_med_desc()
        print("med_code_des: ")
        
    def get_kg(self):
        self.kg_path = os.path.join(self.kg_path, 'kg.csv')
        with open(self.kg_path, 'r') as f:
            kg_df = pd.read_csv(f, low_memory=False)
        print(kg_df.head(10))

        x_index, y_index, rel = kg_df['x_index'], kg_df['y_index'], kg_df['display_relation']
        x_index = [int(i) for i in x_index]
        y_index= [int(i) for i in y_index]
        edge_index = torch.LongTensor([x_index, y_index])
        rel_index = []
        rel_dict = {}
        for r in rel:
            if r not in rel_dict:
               rel_dict[r] = len(rel_dict)
            rel_index.append(rel_dict[r])
        print(edge_index.shape)
        print(rel_dict)
        rel_index = torch.Tensor(rel_index)
        
        return edge_index, rel_index
    
    def get_med_desc(self):
        #pre_path = 'Dataset/medicalCode'
        pre_path = '/n/netscratch/mzitnik_lab/Lab/xsu/MultimodalTokenizer'
        file_name = 'med_code_descriptions.pkl'
        path = os.path.join(pre_path, file_name)
        if os.path.exists(path):
            med_code_des = pd.read_pickle(path)
            return med_code_des
        else:
            med_code_des = {}
            for i in range(len(self.med_codes_pkg_df)):
                med_code = self.med_codes_pkg_df.iloc[i]['med_code']
                desc = self.med_codes_pkg_df.iloc[i]['desc']
                input_ids, attention_mask = self.get_text_tokenizer(desc)
                med_code_des[i] = {'input_ids': input_ids, 'attention_mask': attention_mask}
            pd.to_pickle(med_code_des, path)
        return med_code_des
    
    def get_med_graph(self):
        #pre_path = 'Dataset/medicalCode'
        pre_path = '/n/netscratch/mzitnik_lab/Lab/xsu/MultimodalTokenizer'
        file_name = 'med_code_graph.pkl'
        path = os.path.join(pre_path, file_name)
        if os.path.exists(path):
            med_code_graph = pd.read_pickle(path)
            return med_code_graph
        else:
            med_code_graph = {}
            for i in range(len(self.med_codes_pkg_df)):
                nodes_l = self.med_codes_pkg_df.iloc[i]['pkg_index_list']#subgraph node list
                nodes_l.sort()
                sub_edge_index, sub_rel_index = subgraph(torch.tensor(nodes_l), self.edge_index, self.rel_index, relabel_nodes=True)
                edge_index_aug, rel_index_aug = self.transform(sub_edge_index, sub_rel_index)
                med_code_graph[i] = {'edge_index': sub_edge_index, 'rel_index': sub_rel_index, 'edge_index_aug': edge_index_aug, 'rel_index_aug': rel_index_aug}
            pd.to_pickle(med_code_graph, path)
        return med_code_graph
    
    def get_data(self, idx):
       
        nodes_l = self.med_codes_pkg_df.iloc[idx]['pkg_index_list']#subgraph node list
        nodes_l.sort()

        med_code = self.med_codes_pkg_df.iloc[idx]['med_code']  ##med code, str
        
        input_ids = self.med_code_des[idx]['input_ids']
        attention_mask = self.med_code_des[idx]['attention_mask']

        sub_edge_index, sub_rel_index = subgraph(torch.tensor(nodes_l), self.edge_index, self.rel_index, relabel_nodes=True)
        edge_index_aug, rel_index_aug = self.transform(sub_edge_index, sub_rel_index)
        

        data = DATA.Data(x=torch.LongTensor(nodes_l),  ##medcode idx
                         input_ids = input_ids,  ##medcode tokens shape: (1, max_length)]
                         attention_mask = attention_mask,  ##medcode tokens
                         edge_index = sub_edge_index,
                         rel_index = sub_rel_index,
                         #input_ids_aug = input_ids_aug,
                         #attention_mask_aug = attention_mask_aug,
                         edge_index_aug = edge_index_aug,
                         rel_index_aug = rel_index_aug,
                         code_indices = torch.LongTensor([idx])
                        )
        
        return data
    
    def get_text_tokenizer(self, text):
        encoded = self.text_tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return input_ids, attention_mask
    
    def __len__(self):
        return len(self.med_codes_pkg_df)

    def __getitem__(self, idx):
        return self.get_data(idx)

    def get_subgraph_hetro(self, nodes_l):
        nodes_group_df = self.unique_nodes_df[self.unique_nodes_df['node_index'].isin(nodes_l)]
        selected_nodes_groups = nodes_group_df.groupby('node_type')
        
        filter_dict = {}
        for node_type, split_subset in selected_nodes_groups:
            filter_dict[node_type] = split_subset['node_type_graph_index'].tolist()
    
        sg = dgl.node_subgraph(self.g, filter_dict) # relabel_nodes=False will keep all the nodes from the original kg.
        return sg
        

def custom_collate_fn(data_list):
    batch_data = Batch.from_data_list(data_list)
    return batch_data


if __name__ == "__main__":
    kg_path = '../Dataset/primeKG/kg.csv'
    med_codes_pkg_map_path = '../Dataset/medicalCode/all_codes_mappings.parquet'
    graph_save_path = '../Dataset/medicalCode/kg_temp_2912'

    medcode_dataset = MedCodeDataset(kg_path, graph_save_path, med_codes_pkg_map_path)
    dataloader = DataLoader(medcode_dataset, batch_size=1, num_workers=0, collate_fn=custom_collate_fn)

    for med_codes, sgs in tqdm(dataloader):
        print(med_codes)
        print(sgs)
        break







