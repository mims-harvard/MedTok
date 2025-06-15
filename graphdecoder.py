import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralGraphDecoder(nn.Module):
    def __init__(self, graph_dim, num_nodes, rank):
        super(SpectralGraphDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.rank = rank

        self.fc_eigenvectors = nn.Linear(graph_dim, num_nodes * rank)  # 解码特征向量
        self.fc_eigenvalues = nn.Linear(graph_dim, rank)  # 解码特征值

    def forward(self, q):
        """
        Args:
            q: 图级别的表示, 大小为 [1, d_q]

        Returns:
            adjacency_matrix: 解码出的邻接矩阵, 大小为 [N, N]
        """
        # 解码特征向量 [N, d]
        U = self.fc_eigenvectors(q).view(self.num_nodes, self.rank)

        # 解码特征值 [d]
        Lambda = torch.diag(self.fc_eigenvalues(q))  # 转化为对角矩阵

        # 拉普拉斯矩阵
        L = U @ Lambda @ U.T  # L = U * Λ * U^T

        # 邻接矩阵近似重建
        D = torch.diag(L.sum(dim=-1))  # 度矩阵
        adjacency_matrix = F.relu(torch.inverse(D.sqrt()) @ (torch.eye(self.num_nodes) - L) @ torch.inverse(D.sqrt()))

        return adjacency_matrix

class GraphLevelDecoder(nn.Module):
    def __init__(self, graph_dim, num_nodes, hidden_dim):
        """
        初始化图级别解码器
        Args:
            graph_dim: 图表示向量的维度
            num_nodes: 图中节点数量
            hidden_dim: 中间隐藏维度
        """
        super(GraphLevelDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(graph_dim, num_nodes * hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_nodes)

    def forward(self, q):
        """
        前向传播
        Args:
            q: 图级别表示向量 (batch_size, graph_dim)
        Returns:
            A_pred: 预测的邻接矩阵 (batch_size, num_nodes, num_nodes)
        """
        batch_size = q.size(0)
        
        # 将图表示向量解码为节点嵌入
        node_embeddings = self.fc1(q).view(batch_size, -1, self.hidden_dim)  # (batch_size, num_nodes, hidden_dim)

        # 计算节点之间的关系，生成邻接矩阵
        A_pred = torch.matmul(node_embeddings, node_embeddings.transpose(1, 2))  # (batch_size, num_nodes, num_nodes)
        A_pred = torch.sigmoid(A_pred)  # 映射到 [0, 1]

        return A_pred
