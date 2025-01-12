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
class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        A single cross-attention layer.
        
        Args:
            embed_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for attention.
        """
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        """
        Forward pass of the cross-attention layer.

        Args:
            query (Tensor): Query embeddings (L1, B, D).
            key (Tensor): Key embeddings (L2, B, D).
            value (Tensor): Value embeddings (L2, B, D).
            attn_mask (Tensor, optional): Attention mask. Default is None.

        Returns:
            Tensor: Output of cross-attention (L1, B, D).
        """
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        
        # Add & norm
        query = query + self.dropout(attn_output)
        query = self.layer_norm(query)
        return query

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, layers=2):
        """
        Three-layer cross-attention module.

        Args:
            embed_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for attention.
        """
        super().__init__()
        self.model = nn.ModuleList([CrossAttentionLayer(embed_dim, num_heads, dropout) for _ in range(layers)])

    def forward(self, vector1, vector2, attn_mask=None):
        """
        Forward pass of the three-layer cross-attention module.

        Args:
            vector1 (Tensor): First input vector (L1, B, D).
            vector2 (Tensor): Second input vector (L2, B, D).
            attn_mask (Tensor, optional): Attention mask. Default is None.

        Returns:
            Tuple[Tensor, Tensor]: Updated vector1 and vector2.
        """
        # Cross-attention from vector1 to vector2
        vector1_ = vector1
        vector2_ = vector2

        for layer in self.model:
            vector1_ = layer(vector1_, vector2, vector2, attn_mask)
        
        for layer in self.model:
            vector2_ = layer(vector2_, vector1, vector1, attn_mask)
            
        return vector1_, vector2_

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage, split, kmeans=False, num_head=4, k=5):
        super().__init__()
        self.n_e = n_e  ## number of embeddings, the size of codebook
        self.e_dim = e_dim  ## dimension of each embedding
        self.beta = beta ## weight for commitment loss
        self.entropy_loss_ratio = entropy_loss_ratio ## weight for entropy loss
        self.l2_norm = l2_norm ## whether to normalize the embeddings
        self.show_usage = show_usage ## whether to show the usage of the codebook
        self.split = split ## split the input into two parts, one for text and one for graph
        self.k = k

        self.kmeans_init = kmeans
        self.initted = False

        self.cross_attn = CrossAttention(e_dim, num_head, dropout=0.1, layers=2)
        self.proj_text = nn.Linear(self.split[0], e_dim)
        self.proj_graph = nn.Linear(self.split[1], e_dim)
        
        if self.kmeans_init: # default not use
            print("using kmeans init")
            self.codebook = EmbeddingEMA(self.n_e, self.split[0])
        else:
            print("no kmeans init")
            ##only one codebook, if we only have onecodebook, the n_e should be larger
            self.codebook = nn.Embedding(self.n_e, self.e_dim)

        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(300000)))
    
    def get_distance(self, x, y):
        d = torch.sum(x ** 2, dim=1, keepdim=True) + \
                torch.sum(y**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn',x, torch.einsum('n d -> d n', y))
        
        return d
    
    def get_shared_info(self, z_text, z_graph, text_mask, batch):
        ##the input z_text size [bz, n_tokens, dim], z_graph size [bz, n_nodes, dim], is the output of the encoder before pooling

        z_flatten_text = []
        z_flatten_graph = []
        
        for idx in range(z_text.size(0)):
            text_mask_idx = text_mask[idx, :]
            z_text_idx = z_text[idx, :torch.sum(text_mask_idx, dim=-1).item(), :]
            z_graph_idx = z_graph[(batch == idx).flatten()]
            z_text_attn, z_graph_attn = self.cross_attn(z_text_idx, z_graph_idx)

            z_mapped_text = z_text_attn[0, :] ##0 denote the [CLS] token
            z_mapped_graph = torch.mean(z_graph_attn, dim=0) ##size [bz, dim]
            z_flatten_text.append(z_mapped_text)
            z_flatten_graph.append(z_mapped_graph)
        
        z_flattened_text = torch.stack(z_flatten_text, dim=0)
        z_flattened_graph = torch.stack(z_flatten_graph, dim=0)
    
        if self.l2_norm:
            codebook_embedding_norm = F.normalize(self.codebook.weight, p=2, dim=-1)

            z_flattened_text_norm = F.normalize(z_flattened_text, p=2, dim=-1)
            z_flattened_graph_norm = F.normalize(z_flattened_graph, p=2, dim=-1)

        ##compute the distance between the embeddings and the codebook
        d_text = self.get_distance(z_flattened_text_norm, codebook_embedding_norm)
        d_graph = self.get_distance(z_flattened_graph_norm, codebook_embedding_norm)
        
        values_text, min_encoding_indices_text = torch.topk(d_text, k=self.k, largest=False)
        weights_text = torch.softmax(-values_text, dim=1) 
        values_graph, min_encoding_indices_graph = torch.topk(d_graph, k=self.k, largest=False)
        weights_graph = torch.softmax(-values_graph, dim=1)

        ##get corresponding quantized embeddings
        #z_q = (weights.unsqueeze(-1) * self.codebook[min_encoding_indices]).sum(dim=1).view(z.shape)
        z_q_text = (weights_text.unsqueeze(-1) * codebook_embedding_norm[min_encoding_indices_text]).sum(dim=1).view(z_flattened_text_norm.shape)
        z_q_graph = (weights_graph.unsqueeze(-1) * codebook_embedding_norm[min_encoding_indices_graph]).sum(dim=1).view(z_flattened_graph_norm.shape)

        # compute loss for embedding
        if self.training:
            vq_loss_text = torch.mean((z_q_text - z_flattened_text.detach()) ** 2) ## new indices should be simialr to the original input
            commit_loss_text = self.beta * torch.mean((z_q_text.detach() - z_flattened_text) ** 2) ## the original input should be similar to the new indices, detach means stop gradients

            vq_loss_graph = torch.mean((z_q_graph - z_flattened_graph.detach()) ** 2) ## new indices should be simialr to the original input
            commit_loss_graph = self.beta * torch.mean((z_q_graph.detach() - z_flattened_graph) ** 2) ## the original input should be similar to the new indices, detach means stop gradients
        else:
            vq_loss_text = torch.tensor(0.0)
            commit_loss_text = torch.tensor(0.0)
            vq_loss_graph = torch.tensor(0.0)
            commit_loss_graph = torch.tensor(0.0)

        # preserve gradients
        z_q_text = z_flattened_text + (z_q_text - z_flattened_text).detach()
        z_q_graph = z_flattened_graph + (z_q_graph - z_flattened_graph).detach()
        codebook_usage = self.codebook_usage(torch.concat([min_encoding_indices_text, min_encoding_indices_graph], dim=-1), types='shared')

        return torch.concat([z_q_text, z_q_graph], dim=-1), (vq_loss_text + vq_loss_graph, commit_loss_text+commit_loss_graph, z_flattened_text_norm, z_flattened_graph_norm, z_q_text, z_q_graph), codebook_usage
    
    def specific_embedding(self, original_embedding, types = 'text'):
        ##get the specific embedding for text modality
        if types == 'text':
            original_embedding = self.proj_text(original_embedding)
        if types == 'graph':
            original_embedding = self.proj_graph(original_embedding)

        if self.l2_norm:
            region = self.codebook.weight.shape[0] // 3
            original_embedding_norm = F.normalize(original_embedding, p=2, dim=-1)
            if types == 'text':
                embedding_norm = F.normalize(self.codebook.weight[:region], p=2, dim=-1)
            elif types == 'graph':
                embedding_norm = F.normalize(self.codebook.weight[-region:], p=2, dim=-1)
        
        d_specific = self.get_distance(original_embedding_norm, embedding_norm)
        values, min_encoding_indices = torch.topk(d_specific, k=self.k, largest=False)  ##find the index of the closest embedding for each token
        weights = torch.softmax(-values, dim=1)
        z_q = (weights.unsqueeze(-1) * embedding_norm[min_encoding_indices]).sum(dim=1).view(original_embedding.shape)

        if self.training:
            vq_loss = torch.mean((z_q - original_embedding.detach()) ** 2)
            commit_loss = self.beta * torch.mean((z_q.detach() - original_embedding) ** 2)
        else:
            vq_loss = torch.tensor(0.0)
            commit_loss = torch.tensor(0.0)
        
        z_q = original_embedding + (z_q - original_embedding).detach()
        codebook_usage = self.codebook_usage(min_encoding_indices, types=types+'-specific')

        return z_q, (vq_loss, commit_loss, original_embedding_norm, z_q), codebook_usage

    def codebook_usage(self, min_encoding_indices, types='shared'):
        
        min_encoding_indices = min_encoding_indices.view(-1)
        cur_len = min_encoding_indices.shape[0]
        if types == 'shared':
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_used = len(torch.unique(self.codebook_used)) / self.n_e
        elif types == 'text-specific':
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_used = len(torch.unique(self.codebook_used)) / self.n_e
        elif types == 'graph-specific':
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_used = len(torch.unique(self.codebook_used)) / self.n_e

        return codebook_used

    def forward(self, z, text_features, graph_node_features, text_attention_mask, batch, z_aug=None):
        #h, text_features, graph_node_features, text_attention_mask, batch, h_aug

        shared_embedding, shared_embed_loss, shared_codebook_usage = self.get_shared_info(text_features, graph_node_features, text_attention_mask, batch)
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