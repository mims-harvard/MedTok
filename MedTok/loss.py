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


def info_nce_loss(q, k, temperature=0.07):
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


def shared_loss(z1, z2, x1, x2, beta=0.1):
    """
    Compute the shared loss.
    """
    x1_norm = F.normalize(x1, p=2, dim=-1)
    x2_norm = F.normalize(x2, p=2, dim=-1)
    #loss1 = info_nce_loss(z1, z2) - beta * alignment_loss(x1_norm, x2_norm)
    #loss2 = info_nce_loss(z2, z1) - beta * alignment_loss(x2_norm, x1_norm)
    
    return info_nce_loss(z1, z2), alignment_loss(x1_norm, x2_norm), info_nce_loss(z2, z1), alignment_loss(x2_norm, x1_norm)
    #return loss1 + loss2

def specific_loss(z1, z1_aug, z2, z2_aug, z1_c, z2_c, lamb=0.1):
    """
    Compute the specific loss.
    """
    z1_hat = torch.cat([z1, z2_c], dim=-1)
    z1_aug_hat = torch.cat([z1_aug, z2_c], dim=-1)
    loss1 = info_nce_loss(z1_hat, z1_aug_hat) + lamb * orthogonal_loss(z1, z1_c)

    z2_hat = torch.cat([z2, z1_c], dim=-1)
    z2_aug_hat = torch.cat([z2_aug, z1_c], dim=-1)
    loss2 = info_nce_loss(z2_hat, z2_aug_hat) + lamb * orthogonal_loss(z2, z2_c)

    return info_nce_loss(z1_hat, z1_aug_hat), orthogonal_loss(z1, z1_c), info_nce_loss(z2_hat, z2_aug_hat), orthogonal_loss(z2, z2_c)



