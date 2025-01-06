import torch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from dgl.nn import GraphConv, GATConv
import os
os.environ['HF_HOME'] = '/n/netscratch/mzitnik_lab/Lab/xsu/xiaorui/cache/'
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from timm.models.layers import trunc_normal_
from vector_quantization import VectorQuantizer
import random
import dgl
import torch.nn.functional as F
from graphdecoder import SpectralGraphDecoder, GraphLevelDecoder
from loss import edge_loss

class GraphEncoder(torch.nn.Module):
    def __init__(self, model_name, in_channels, hidden_channels, out_channels, num_nodes):
        super(GraphEncoder, self).__init__()
        self.model_name = model_name
        self.emb = nn.Embedding(num_nodes, in_channels)
        if self.model_name == "GCN":
            self.model = nn.Sequential(
                GCNConv(in_channels, hidden_channels),
                nn.ReLU(),
                GCNConv(hidden_channels, out_channels)
            )
        elif self.model_name == "GAT":
            self.num_heads = 4
            self.model = nn.Sequential(
                GATConv(in_channels, hidden_channels, num_heads = self.num_heads),
                nn.ReLU(),
                GATConv(hidden_channels * self.num_heads, out_channels, 1)
            )
        else:
            raise ValueError("Invalid graph model name. Please choose from 'GCN' or 'GAT'.")

    def forward(self, indices, edge_index, rel_index):
        
        h = []
        x = self.emb(indices)
        for idx, layer in enumerate(self.model):
            if idx % 2 == 0:
                x = layer(x, edge_index)
                h.append(x)
            else:
                x = layer(x)

        return h ##return the hidden states of each layer

def drop_scale(original_scales, num_to_drop=1):
    """
    Randomly remove scales from scale list.
    
    Args:
        original_scales: list of scales
        num_to_drop: Number of scales to randomly remove (default 1)
        
    Returns:
        New scale list
    """
    if num_to_drop >= len(original_scales) - 1:
        raise ValueError("Cannot drop that many items")
    
    drop_candidates = list(range(1, len(original_scales)))
    indices_to_drop = set(random.sample(drop_candidates, num_to_drop))
    return [item for i, item in enumerate(original_scales) if i not in indices_to_drop]


class MultimodalTokenizer(torch.nn.Module):
    def __init__(self, text_model_name: str = "bert-base-uncased", graph_model_name: str = "GCN",
                 graph_in_channels: int = 128, graph_hidden_channels: int = 64, graph_out_channels: int = 32,
                 encoder_text_code_dim: int = 256, decoder_dim: int = 256, encoder_dim: int = 256, output_channels: int = 128,
                 encoder_decoder_hidden_states: int = 256,
                 codebook_size: int = 18000, codebook_embed_dim: int = 8, commit_loss_beta: float = 1.0, entropy_loss_ratio: float = 0.0,
                 codebook_l2_norm: bool = True, codebook_show_usage: bool = True, use_kmeans: bool = False):
        super(MultimodalTokenizer, self).__init__()
        # Initialize text tokenizer and model
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Initialize graph encoder
        self.graph_encoder = GraphEncoder(model_name=graph_model_name, in_channels=graph_in_channels, hidden_channels=graph_hidden_channels, out_channels=graph_out_channels, num_nodes=130000)

        self.text_code_dim = encoder_text_code_dim
        self.hidden_states = encoder_decoder_hidden_states
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim


        self.codebook_size = codebook_size
        self.code_dim = codebook_embed_dim
        self.commit_loss_beta = commit_loss_beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.codebook_l2_norm = codebook_l2_norm
        self.codebook_show_usage = codebook_show_usage


        # Task Layer
        self.encoder_task_layer = nn.Sequential(
            nn.Linear(graph_out_channels, self.hidden_states),
            nn.Tanh(),
            nn.Linear(self.hidden_states, self.text_code_dim) ##for quantize??
        )
        self.decoder_task_layer = nn.Sequential(
            nn.Linear(self.decoder_dim, self.hidden_states),
            nn.Tanh(),
            nn.Linear(self.hidden_states, output_channels)
        )
        self.encoder_task_layer.apply(self._init_weights)
        self.decoder_task_layer.apply(self._init_weights)
        ##here the decoder tasks, one should be the structure reconstruction and the node type reconstruction, the other should be what??
        self.graph_decoder = GraphLevelDecoder(graph_dim=graph_out_channels, num_nodes=2000, hidden_dim=graph_hidden_channels)

        ##parameters required for vector quantization
        self.text_code_dim = 768 ##the dim of text encoder
        self.vqgraph_code_dim = codebook_embed_dim
        self.code_dim = self.text_code_dim + self.vqgraph_code_dim

        self.embed_dim = self.code_dim
        self.n_embed = codebook_size  ###quantized size
        self.compression = 2**(self.codebook_size-1)

        ## quantizer
        ## semantic_code_dim is the dim of text encoder
        ## vqgan_code_dim is the dim of graph encoder
        self.use_kmeans = use_kmeans
        self.quantize = VectorQuantizer(n_e=self.codebook_size, e_dim=self.code_dim, 
                                beta=self.commit_loss_beta, entropy_loss_ratio=self.entropy_loss_ratio,
                                l2_norm=self.codebook_l2_norm, show_usage=self.codebook_show_usage, split=[self.text_code_dim, self.vqgraph_code_dim], kmeans=self.use_kmeans)

        ### whether using random scale drop when training ###
        self.random_scale_drop = True
        self.random_scale_drop_ratio = 0.1 # default 10%
        self.num_of_random_drop = 1
        self.enable_var = False
        self.lamb_edge = 1.0
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def tokenize_text(self, input_ids, attention_mask):
        """Tokenizes and encodes the text input."""
        with torch.no_grad():
            outputs = self.text_model(input_ids, attention_mask)  ##size:[baz, token_len, out_channels] the out_channels depends on the backbone model
        #print("tokenize_text: ", outputs.last_hidden_state.shape)   
        return outputs.last_hidden_state.mean(dim=1)  # Return sentence-level representation

    def tokenize_graph(self, x, edge_index, rel_index):
        """Encodes the graph input using a graph neural network."""
        return self.graph_encoder(x, edge_index, rel_index)
    
    def quant(self, text_features, graph_features):
        ##input: text_features: [bz, text_dim], graph_features: [bz, graph_dim]
        h = torch.cat((text_features, graph_features), dim=-1)  ##[bz, text_dim + graph_dim]
        #print("quant_input: ", h.shape)

        if self.enable_var:
            # VAR
            residual = h
            final_quantize = 0.
            all_loss = []
            all_inds = []

            new_layers = self.scale_rq_layers
            if self.training and self.random_scale_drop:
                if random.random() < self.random_scale_drop_ratio:
                    new_layers = drop_scale(self.scale_rq_layers, self.num_of_random_drop)

            for i, scale_size in enumerate(new_layers): # random drop scale
                residual_si = F.interpolate(residual, size=(scale_size, scale_size), mode='area')
                quantize, emb_loss, info = self.quantize(residual_si)
                quantize = F.interpolate(quantize, size=(self.scale_rq_layers[-1], self.scale_rq_layers[-1]), mode='bicubic')

                final_quantize += quantize
                residual = residual - quantize.detach()

                all_loss.append(emb_loss)
                inds = info[-1].reshape([-1, scale_size ** 2])
                all_inds.append(inds)

            all_loss = zip(*all_loss)
            all_loss = [sum(group) / len(group) if group[0] is not None else None for group in all_loss]
            all_inds = torch.cat(all_inds, dim=1)

            return final_quantize, all_loss, all_inds
        else:
            quant, emb_loss, info = self.quantize(h)
            return quant, emb_loss, info
    
    '''def decode(self, quant):
        vqkd_quant, vqgan_quant = torch.split(quant, [self.semantic_code_dim, self.vqgan_code_dim], dim=1)
        vqkd_recon = self.decoder_vqkd(vqkd_quant, return_patch_tokens=True)
        vqkd_recon = self.decode_task_layer(vqkd_recon)

        vqgan_recon = self.post_quant_conv(vqgan_quant)
        vqgan_recon = self.decoder(vqgan_recon)

        if self.teacher == 'siglip_384':
            vqgan_recon = F.interpolate(vqgan_recon, size=(384, 384), mode='bicubic')

        dec = (vqkd_recon, vqgan_recon)
        return dec'''
    
    def decode(self, quant):
        text_quant, graph_quant = torch.split(quant, [self.text_code_dim, self.vqgraph_code_dim], dim=1)

        ##decode the graph quant
        recon_adj = self.graph_decoder(graph_quant)

        return recon_adj

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec
    
    def forward(self, inputs):
        ## the input is (medical code description, medical code graph)
        ## text_input_ids [bz, max_length], text_attention_mask [bz, max_length], graph_edge_index [2, edge_num], graph_rel_index [edge_num]
        
        ##encode the text
        text_input_ids, text_attention_mask, graph_edge_index, graph_rel_index = inputs.input_ids, inputs.attention_mask, inputs.edge_index, inputs.rel_index
        text_features = self.tokenize_text(text_input_ids, text_attention_mask)  ##[bz, max_length, text_dim]
        
        ##graph encoder
        graph_hidden_states = self.tokenize_graph(inputs.x, graph_edge_index, graph_rel_index) ##[bz, node_num, graph_dim], returned feature is a list recording the hidden states of each layer
        graph_features = global_mean_pool(graph_hidden_states[-1], inputs.batch) ##use the last layer hidden states as the graph features
        

        ##mapped text_features and graph_features to the same codebook
        #first mapped text_features to the same codebook
        ## z_q, 
        ## (vq_loss, commit_loss, entropy_loss, codebook_usage, text_d_norm, graph_d_norm, z_q_text, z_q_graph), 
        ## (perplexity, min_encodings, min_encoding_indices)
        ## quant: [bz, code_dim], embed_loss is a list of loss, _ is a list of indices
        quant, embed_loss, _ = self.quant(text_features, graph_features)

        ##decode the quantized features to make sure the quantization is correct and not loss too much information
        recon_adj = self.decode(quant)

        '''num_nodes = inputs.x.shape[0]
        print("graph_edge_index: ", graph_edge_index.device)
        values = torch.ones(graph_edge_index.size(1), device=graph_edge_index.device)
        adj_dense = torch.sparse_coo_tensor(graph_edge_index, values, (num_nodes, num_nodes)).to(graph_edge_index.device).to_dense()
        print(adj_dense.shape)
        print(recon_adj.shape)
        edge_rec_loss = self.lamb_edge * torch.sqrt(F.mse_loss(adj_dense, recon_adj))'''

        return recon_adj, embed_loss, 0.0

        #return dec, emb_loss

    def tokenize(self, text=None, node_features=None, edge_index=None):
        """Tokenizes and combines text and graph inputs into a multimodal representation."""
        if text is not None:
            text_features = self.tokenize_text(text)
        else:
            text_features = None

        if node_features is not None and edge_index is not None:
            graph_features = self.tokenize_graph(node_features, edge_index)
        else:
            graph_features = None

        # Combine text and graph features (simple concatenation as an example)
        if text_features is not None and graph_features is not None:
            multimodal_features = torch.cat((text_features, graph_features.unsqueeze(0)), dim=1)
        elif text_features is not None:
            multimodal_features = text_features
        elif graph_features is not None:
            multimodal_features = graph_features.unsqueeze(0)
        else:
            raise ValueError("At least one of text or graph input must be provided.")

        return multimodal_features

# Example usage
if __name__ == "__main__":#
    tokenizer = MultimodalTokenizer()

    # Text input (medical code description)
    text_input = "Hypertension is a condition with elevated blood pressure."

    # Graph input (biological process subgraph)
    node_features = torch.rand((10, 128))  # 10 nodes with 128-dimensional features
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])  # Edge list in COO format

    # Tokenize and combine
    multimodal_output = tokenizer.tokenize(text=text_input, node_features=node_features, edge_index=edge_index)
    print("Multimodal features shape:", multimodal_output.shape)

