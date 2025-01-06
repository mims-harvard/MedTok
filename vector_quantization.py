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
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(131072)))

    def forward(self, z):
        ## input shape: (b, text_dim + graph_dim)
        z_text, z_graph = torch.split(z, split_size_or_sections=self.split, dim=-1) ##graph_text: [bz, node_num, text_dim], graph_graph: [bz, node_num, graph_dim]
        
        z_flattened = z.view(-1, self.e_dim)  ##[bz, text_dim + graph_dim]
        z_flattened_text, z_flattened_graph = torch.split(z_flattened, split_size_or_sections=self.split, dim=-1) ##graph_flattened_text: [bz*node_num, text_dim], graph_flattened_graph: [bz*node_num, graph_dim]

        if self.l2_norm: ##flattened embeddings are normalized
            z_flattened_text_norm = F.normalize(z_flattened_text, p=2, dim=-1)
            z_flattened_graph_norm = F.normalize(z_flattened_graph, p=2, dim=-1)
            z_flattened_norm = torch.cat([z_flattened_text_norm, z_flattened_graph_norm], dim=-1)
        
        if self.l2_norm: ##embeddings are normalized
            z_text_norm = F.normalize(z_text, p=2, dim=-1)
            embedding_text_norm = F.normalize(self.embedding_text.weight, p=2, dim=-1)  ##codebook for text

            z_graph_norm = F.normalize(z_graph, p=2, dim=-1)
            embedding_graph_norm = F.normalize(self.embedding_graph.weight, p=2, dim=-1)

            z = torch.cat([z_text_norm, z_graph_norm], dim=-1)
        
        ##compute the distance between the embeddings and the codebook
        d_text = torch.sum(z_flattened_text_norm ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding_text_norm**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened_text_norm, torch.einsum('n d -> d n', embedding_text_norm))
        d_graph = torch.sum(z_flattened_graph_norm ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding_graph_norm**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened_graph_norm, torch.einsum('n d -> d n', embedding_graph_norm))
        
        text_d_norm = torch.mean(torch.sum(d_text**2, dim=-1))  ##this is a value
        graph_d_norm = torch.mean(torch.sum(d_graph**2, dim=-1)) ##this is a value

        ### shared mapping ###
        d = d_text + 1.0 * d_graph  ##size [bz, codebook_size]
        
        min_encoding_indices = torch.argmin(d, dim=1)  ##find the index of the closest embedding for each token
        
        all_embedding = torch.cat([embedding_text_norm, embedding_graph_norm], dim=-1)

        ##get corresponding quantized embeddings
        z_q = all_embedding[min_encoding_indices].view(z.shape)
        z_q_text = embedding_text_norm[min_encoding_indices].view(z_text_norm.shape)
        z_q_graph = embedding_graph_norm[min_encoding_indices].view(z_graph_norm.shape)

        aggregate_usage = False
        if aggregate_usage:
            with torch.no_grad():
                min_encoding_indices_all = [torch.zeros_like(min_encoding_indices) for _ in range(torch.distributed.get_world_size())]
                dist.all_gather(min_encoding_indices_all, min_encoding_indices)
                min_encoding_indices_all = torch.cat(min_encoding_indices_all, dim=0)
        
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            if aggregate_usage:
                cur_len = min_encoding_indices_all.shape[0]
                self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
                self.codebook_used[-cur_len:] = min_encoding_indices_all
                codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e
            else:
                cur_len = min_encoding_indices.shape[0]
                self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
                self.codebook_used[-cur_len:] = min_encoding_indices
                codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e
        
        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) ## new indices should be simialr to the original input
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) ## the original input should be similar to the new indices, detach means stop gradients
            #print("selected indices", d)
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

            ##should add two construction loss for graph

        # preserve gradients
        z_q = z + (z_q - z).detach()
        #print("z_q", z_q.shape)
        #print("vq_loss", vq_loss)
        #print("commit_loss", commit_loss)
        #print("entropy_loss", entropy_loss)
        #print("codebook_usage", codebook_usage)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage, text_d_norm, graph_d_norm, z_q_text, z_q_graph), (perplexity, min_encodings, min_encoding_indices)

        
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