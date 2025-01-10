import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import numpy as np
from transformers.modeling_utils import get_parameter_device, get_parameter_dtype
from norm_ema_quantizer import EmbeddingEMA, l2norm, norm_ema_inplace
import torch.distributed as dist
from graphdecoder import SpectralGraphDecoder
##soft means that the embedding are decided by top 10 closest embeddings, not the minimum one
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage, split, kmeans=False):
        super().__init__()
        self.n_e = n_e  ## number of embeddings, the size of codebook
        self.e_dim = e_dim  ## dimension of each embedding
        self.beta = beta ## weight for commitment loss
        self.entropy_loss_ratio = entropy_loss_ratio ## weight for entropy loss
        self.l2_norm = l2_norm ## whether to normalize the embeddings
        self.show_usage = show_usage ## whether to show the usage of the codebook
        self.split = split ## split the input into two parts, one for text and one for graph

        self.kmeans_init = kmeans
        self.initted = False
        
        if self.kmeans_init: # default not use
            print("using kmeans init")
            self.embedding_text = EmbeddingEMA(self.n_e, self.split[0])
            self.embedding_text.weight.requires_grad = False

            self.embedding_graph = EmbeddingEMA(self.n_e, self.split[1])
            self.embedding_graph.weight.requires_grad = False
        else:
            print("no kmeans init")
            self.embedding_text = nn.Embedding(self.n_e, self.split[0])  ## embedding for text
            self.embedding_text.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
            if self.l2_norm:
                self.embedding_text.weight.data = F.normalize(self.embedding_text.weight.data, p=2, dim=-1)

            self.embedding_graph = nn.Embedding(self.n_e, self.split[1]) ## embedding for graph
            self.embedding_graph.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
            if self.l2_norm:
                self.embedding_graph.weight.data = F.normalize(self.embedding_graph.weight.data, p=2, dim=-1)

            self.shared_embedding_text = nn.Embedding(self.n_e, self.split[0]) ## shared embedding for text
            self.shared_embedding_text.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
            if self.l2_norm:
                self.shared_embedding_text.weight.data = F.normalize(self.shared_embedding_text.weight.data, p=2, dim=-1)
            
            self.shared_embedding_graph = nn.Embedding(self.n_e, self.split[1]) ## shared embedding for graph
            self.shared_embedding_graph.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
            if self.l2_norm:
                self.shared_embedding_graph.weight.data = F.normalize(self.shared_embedding_graph.weight.data, p=2, dim=-1)

        if self.show_usage:
            self.register_buffer("shared_codebook_used", nn.Parameter(torch.zeros(100000)))
            self.register_buffer("text_codebook_used", nn.Parameter(torch.zeros(100000)))
            self.register_buffer("graph_codebook_used", nn.Parameter(torch.zeros(100000)))
    
    def get_distance(self, x, y):
        d = torch.sum(x ** 2, dim=1, keepdim=True) + \
                torch.sum(y**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn',x, torch.einsum('n d -> d n', y))
        
        return d
    
    def get_shared_info(self, z):
        ##get the shared info between text and graph modality
        z_flattened_text, z_flattened_graph = torch.split(z, split_size_or_sections=self.split, dim=-1) 

        if self.l2_norm:
            z_flattened_text_norm = F.normalize(z_flattened_text, p=2, dim=-1)
            shared_embedding_text_norm = F.normalize(self.shared_embedding_text.weight, p=2, dim=-1)

            z_flattened_graph_norm = F.normalize(z_flattened_graph, p=2, dim=-1)
            shared_embedding_graph_norm = F.normalize(self.shared_embedding_graph.weight, p=2, dim=-1)

            shared_embedding = torch.cat([shared_embedding_text_norm, shared_embedding_graph_norm], dim=-1)

        ##compute the distance between the embeddings and the codebook
        d_text = self.get_distance(z_flattened_text_norm, shared_embedding_text_norm)
        d_graph = self.get_distance(z_flattened_graph_norm, shared_embedding_graph_norm)
        
        ### shared mapping ###
        d = d_text + 1.0 * d_graph  ##size [bz, codebook_size]
        
        values, min_encoding_indices = torch.topk(d, k=10, largest=False)
        #min_encoding_indices = torch.argmin(d, dim=1)  ##find the index of the closest embedding for each token
        weights = torch.softmax(-values, dim=1)  # 计算权重

        ##get corresponding quantized embeddings
        #z_q = shared_embedding[min_encoding_indices].view(z.shape)
        #z_q_text = shared_embedding_text_norm[min_encoding_indices].view(z_flattened_text_norm.shape)
        #z_q_graph = shared_embedding_graph_norm[min_encoding_indices].view(z_flattened_graph_norm.shape)
        z_q = (weights.unsqueeze(-1) * shared_embedding[min_encoding_indices]).sum(dim=1).view(z.shape)
        z_q_text = (weights.unsqueeze(-1) * shared_embedding_text_norm[min_encoding_indices]).sum(dim=1).view(z_flattened_text_norm.shape)
        z_q_graph = (weights.unsqueeze(-1) * shared_embedding_graph_norm[min_encoding_indices]).sum(dim=1).view(z_flattened_graph_norm.shape)

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) ## new indices should be simialr to the original input
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) ## the original input should be similar to the new indices, detach means stop gradients

        # preserve gradients
        z_q = z + (z_q - z).detach()
        codebook_usage = self.codebook_usage(min_encoding_indices, types='shared')

        return z_q, (vq_loss, commit_loss, z_flattened_text_norm, z_flattened_graph_norm, z_q_text, z_q_graph), codebook_usage
    
    def specific_embedding(self, original_embedding, types = 'text'):
        ##get the specific embedding for text modality
        if self.l2_norm:
            original_embedding_norm = F.normalize(original_embedding, p=2, dim=-1)
            if types == 'text':
                embedding_norm = F.normalize(self.embedding_text.weight, p=2, dim=-1)
            elif types == 'graph':
                embedding_norm = F.normalize(self.embedding_graph.weight, p=2, dim=-1)
        
        d_specific = self.get_distance(original_embedding_norm, embedding_norm)
        values, min_encoding_indices = torch.topk(d_specific, k=10, largest=False)  ##find the index of the closest embedding for each token
        weights = torch.softmax(-values, dim=1)  # 计算权重
        z_q = (weights.unsqueeze(-1) * embedding_norm[min_encoding_indices]).sum(dim=1).view(original_embedding.shape)

        if self.training:
            vq_loss = torch.mean((z_q - original_embedding.detach()) ** 2)
            commit_loss = self.beta * torch.mean((z_q.detach() - original_embedding) ** 2)
        
        z_q = original_embedding + (z_q - original_embedding).detach()
        codebook_usage = self.codebook_usage(min_encoding_indices, types=types+'-specific')

        return z_q, (vq_loss, commit_loss, original_embedding_norm, z_q), codebook_usage

    def codebook_usage(self, min_encoding_indices, types='shared'):
        
        min_encoding_indices = min_encoding_indices.view(-1)
        cur_len = min_encoding_indices.shape[0]
        if types == 'shared':
            self.shared_codebook_used[:-cur_len] = self.shared_codebook_used[cur_len:].clone()
            self.shared_codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.shared_codebook_used)) / self.n_e
        elif types == 'text-specific':
            self.text_codebook_used[:-cur_len] = self.text_codebook_used[cur_len:].clone()
            self.text_codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.text_codebook_used)) / self.n_e
        elif types == 'graph-specific':
            self.graph_codebook_used[:-cur_len] = self.graph_codebook_used[cur_len:].clone()
            self.graph_codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.graph_codebook_used)) / self.n_e

        return codebook_usage

    def forward(self, z, z_aug = None):
        shared_embedding, shared_embed_loss, shared_codebook_usage = self.get_shared_info(z)
        shared_text_embedding, shared_graph_embedding = torch.split(shared_embedding, split_size_or_sections=self.split, dim=-1)
        z_text_embedding, z_graph_embedding = torch.split(z, split_size_or_sections=self.split, dim=-1) ##graph_text: [bz, node_num, text_dim], graph_graph: [bz, node_num, graph_dim]
        specific_embedding_text, text_specific_loss, text_specific_usage = self.specific_embedding(z_text_embedding, types = 'text')
        specific_embedding_graph, graph_specific_loss, graph_specific_usage = self.specific_embedding(z_graph_embedding, types = 'graph')

        if z_aug is not None:
            z_aug_text, z_aug_graph = torch.split(z_aug, split_size_or_sections=self.split, dim=-1)
            specific_embedding_text_aug, _, _ = self.specific_embedding(z_aug_text, types = 'text')
            specific_embedding_graph_aug, _, _ = self.specific_embedding(z_aug_graph, types = 'graph')
        else:
            specific_embedding_text_aug = None
            specific_embedding_graph_aug = None


        return {
            'graph_feature': z_graph_embedding,
            'text_feature': z_text_embedding,
            'shared_text_embedding': shared_text_embedding,
            'shared_graph_embedding': shared_graph_embedding,
            'shared_embed_loss': shared_embed_loss,
            'shared_codebook_usage': shared_codebook_usage,
            'specific_embedding_text': specific_embedding_text,
            'text_specific_loss': text_specific_loss,
            'text_specific_usage': text_specific_usage,
            'specific_embedding_graph': specific_embedding_graph,
            'graph_specific_loss': graph_specific_loss,
            'graph_specific_usage': graph_specific_usage,
            'specific_embedding_text_aug': specific_embedding_text_aug,
            'specific_embedding_graph_aug': specific_embedding_graph_aug
        }
        
def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):

    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss