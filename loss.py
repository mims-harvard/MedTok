import torch
import torch.nn as nn
import torch.nn.functional as F

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def edge_loss(reconstructed_adj, edge_index, num_nodes, neg_sample_ratio=1.0):
    """
    计算重构邻接矩阵和真实边的损失
    Args:
        reconstructed_adj: 重构的邻接矩阵 (N, N)
        edge_index: 边索引矩阵 (2, E)
        num_nodes: 节点数量 N
        neg_sample_ratio: 负样本采样比例
    Returns:
        loss: 标量，损失值
    """
    # 正样本的预测值
    pos_edges = edge_index  # (2, E)
    pos_pred = reconstructed_adj[pos_edges[0], pos_edges[1]]  # (E,)
    pos_loss = -torch.log(pos_pred + 1e-8).mean()  # 防止 log(0)

    # 负样本采样
    num_neg_samples = int(pos_edges.size(1) * neg_sample_ratio)
    all_edges = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes))

    # 生成负样本
    neg_edges = all_edges[~torch.any((all_edges == edge_index.T).all(dim=1), dim=0)]
    neg_edges = neg_edges[torch.randperm(len(neg_edges))[:num_neg_samples]]
    neg_pred = reconstructed_adj[neg_edges[:, 0], neg_edges[:, 1]]  # (num_neg_samples,)

    neg_loss = -torch.log(1 - neg_pred + 1e-8).mean()

    # 总损失
    loss = pos_loss + neg_loss
    return loss

def info_nce_loss(q, k, temperature=0.07):
    """
    计算InfoNCE损失
    Args:
        q: Q网络输出的特征 (N, D)
        k: K网络输出的特征 (N, D)
        temperature: 温度参数
    Returns:
        loss: 标量，损失值
    """
    N = q.size(0)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    positive_similarities = torch.sum(q * k, dim=-1) / temperature  # (N, 1)
    mask = ~torch.eye(N, device=q.device).bool()
    negative_similarities = torch.matmul(q, k.t()) / temperature  # (N, N)
    negative_similarities_with_mask = negative_similarities[mask].view(N, -1)

    # Concatenate positive and negative similarities for InfoNCE loss
    logits = torch.cat([positive_similarities.unsqueeze(1), negative_similarities_with_mask], dim=-1)  # Shape: (batch_size, 1 + num_negatives)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=q.device)  # Positives are at index 0

    contrastive_loss = F.cross_entropy(logits, labels)

    return contrastive_loss

# Regularization Term
def alignment_loss(mu1, mu2):
    """
    Compute alignment term: E[x1, x2] mu(x1)^T mu(x2).
    """
    similarity = torch.mean(torch.sum(mu1 * mu2, dim=1))  # Dot product for each pair, then average
    return similarity

def orthogonal_loss(z, z_star):
    """
    Computes the orthogonal loss to encourage independence between Z1 and Z1_star.

    Args:
        Z1 (torch.Tensor): Shared representations, shape (B, d1).
        Z1_star (torch.Tensor): Modality-specific representations, shape (B, d2).

    Returns:
        torch.Tensor: Orthogonal loss value.
    """
    # Compute pairwise dot product (B, d1) @ (B, d2).T -> (d1, d2)
    similarity_matrix = torch.mm(z.T, z_star)
    
    # Frobenius norm of the similarity matrix
    frobenius_norm = torch.norm(similarity_matrix, p='fro')
    
    return frobenius_norm


def shared_loss(z1, z2, x1, x2, beta=0.5):
    """
    Compute the shared loss.
    """
    loss1 = info_nce_loss(z1, z2) - beta * alignment_loss(x1, x2)
    loss2 = info_nce_loss(z2, z1) - beta * alignment_loss(x2, x1)
    return loss1 + loss2

def specific_loss(z1, z1_aug, z2, z2_aug, z1_c, z2_c, lamb=0.5):
    """
    Compute the specific loss.
    """
    z1_hat = torch.cat([z1, z2_c], dim=-1)
    z1_aug_hat = torch.cat([z1_aug, z2_c], dim=-1)
    loss1 = info_nce_loss(z1_hat, z1_aug_hat) + lamb * orthogonal_loss(z1, z1_c)

    z2_hat = torch.cat([z2, z1_c], dim=-1)
    z2_aug_hat = torch.cat([z2_aug, z1_c], dim=-1)
    loss2 = info_nce_loss(z2_hat, z2_aug_hat) + lamb * orthogonal_loss(z2, z2_c)

    return loss1 + loss2


class MultimodalTokenizerLoss(nn.Module):
    def __init__(self, reconstruction_loss='l2', reconstruction_weight=1.0, 
                 codebook_weight=1.0
    ):
        super().__init__()

        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss
        self.codebook_weight = codebook_weight
        self.embedding = nn.Embedding(100, 512)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def calculate_clip_rec_loss(self, rec, target):  
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        return rec_loss
    
    def edge_loss(self, reconstructed_adj, edge_index, num_nodes, neg_sample_ratio=1.0):
        """
        计算重构邻接矩阵和真实边的损失
        Args:
            reconstructed_adj: 重构的邻接矩阵 (N, N)
            edge_index: 边索引矩阵 (2, E)
            num_nodes: 节点数量 N
            neg_sample_ratio: 负样本采样比例
        Returns:
            loss: 标量，损失值
        """
        # 正样本的预测值
        pos_edges = edge_index  # (2, E)
        pos_pred = reconstructed_adj[pos_edges[0], pos_edges[1]]  # (E,)
        pos_loss = -torch.log(pos_pred + 1e-8).mean()  # 防止 log(0)

        # 负样本采样
        num_neg_samples = int(pos_edges.size(1) * neg_sample_ratio)
        all_edges = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes))
        all_edges = all_edges.to(edge_index.device)

        # 生成负样本
        neg_edges = all_edges[~torch.any((all_edges == edge_index.T).all(dim=1), dim=0)]
        neg_edges = neg_edges[torch.randperm(len(neg_edges))[:num_neg_samples]]
        neg_pred = reconstructed_adj[neg_edges[:, 0], neg_edges[:, 1]]  # (num_neg_samples,)

        neg_loss = -torch.log(1 - neg_pred + 1e-8).mean()

        # 总损失
        loss = pos_loss + neg_loss
        return loss

    
    def forward(self, codebook_loss, inputs, reconstructed_adj, global_step, 
                logger=None, log_every=100):

        graph_recon = reconstructed_adj
        
        if graph_recon is not None:
            edge_index, num_nodes = inputs.edge_index, inputs.num_nodes
        
        rec_loss = self.edge_loss(graph_recon, edge_index, num_nodes)
        loss = self.rec_weight * rec_loss + \
            codebook_loss[0] + codebook_loss[1] + codebook_loss[2] + codebook_loss[3]
        
        if global_step % log_every == 0:
            rec_loss = self.rec_weight * rec_loss
            logger.info(f"(Generator) rec_loss: {rec_loss:.4f}"
                        f"vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, entropy_loss: {codebook_loss[2]:.4f}, "
                        f"codebook_usage: {codebook_loss[3]:.4f},"
                        f"d_vqkd: {codebook_loss[4]:.4f}, d_vqgan: {codebook_loss[5]:.4f}")

        loss_dict = {
            'rec_loss': rec_loss,
            'vq_loss': codebook_loss[0],
            'commit_loss': codebook_loss[1],
            'entropy_loss': codebook_loss[2],
            'codebook_usage': codebook_loss[3],
            'd_text': codebook_loss[4],
            'd_graph': codebook_loss[5],
        }
            
        return loss, loss_dict
