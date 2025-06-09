'''
HETEROGENEOUS GRAPH TRANSFROMER
We define a heterogeneous graph transformer model to learn node embeddings on the knowledge graph.
'''

# Import PyTorch
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

# Import DGL
import dgl
# from dgl.dataloading import ShaDowKHopSampler

# Import PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningModule as LM
from torch_geometric.nn import GCNConv
# Path manipulation
from pathlib import Path
import math
from torch import Tensor
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Import project config file
import sys
# sys.path.append('../..')
# import project_config

# Check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from torch_geometric.nn import global_mean_pool

import wandb


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, max_year: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.year_emb = nn.Embedding(max_year, d_model-4)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def get_time_embedding(self, time: Tensor) -> Tensor:
        year = time[:, :, 0]
        day = time[:, :, 1]
        hour = time[:, :, 2]

        day_sin = torch.sin(2 * math.pi * day / 365).unsqueeze(-1)
        day_cos = torch.cos(2 * math.pi * day / 365).unsqueeze(-1)

        hour_sin = torch.sin(2 * math.pi * hour / 24).unsqueeze(-1)
        hour_cos = torch.cos(2 * math.pi * hour / 24).unsqueeze(-1)

        time_embedding = self.year_emb(year)
    
        return torch.cat([time_embedding, day_sin, day_cos, hour_sin, hour_cos], dim=-1)

    def forward(self, x: Tensor, time_within_visit: Tensor, time_between_visit: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        #print("x", x.shape)
        #print("time_within_visit", time_within_visit.shape)
        #print("time_between_visit", time_between_visit.shape)
        time_within_visit_emb = self.get_time_embedding(time_within_visit)
        time_between_visit_emb = self.get_time_embedding(time_between_visit)
        #print("pe",self.pe[:x.size(1)].squeeze(1).shape)
        x = x + self.pe[:x.size(1)].squeeze(1) + time_within_visit_emb + time_between_visit_emb

        return self.dropout(x)

# HETEROGENEOUS GRAPH TRANSFORMER
class EHRModel(pl.LightningModule):
    def __init__(self, model_name, input_dim, num_feat=128, num_heads=4,
                 hidden_dim=256, output_dim=128, num_layers=3, max_visit_num=50,
                 dropout_prob=0.5, pred_threshold=0.5, max_ehr_length=6000, code_size=600000,
                 lr=0.0001, task = 'readmission', wd=0.0, lr_factor=0.01, num_class = 2, lr_patience=100, lr_threshold=1e-4,
                 lr_threshold_mode='rel', lr_cooldown=0, min_lr=1e-8, eps=1e-8, lr_total_iters=10, memory_bank_size = 512, hparams = None):

        super().__init__()

        self.model_name = model_name
        if self.model_name == 'Transformer':
            self.model = nn.ModuleList([
                nn.TransformerEncoderLayer(
                d_model=input_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim, 
                dropout=dropout_prob
            ) 
            for _ in range(num_layers)
        ])

        # Do not cache KG unless requested by user in cache_graph()
        self.cached_kg = None
        self.cached_degree_threshold = None

        # Save model hyperparameters
        # self.save_hyperparameters(hparams)
        self.num_feat = num_feat #hparams['num_feat']
        self.num_heads =  num_heads #hparams['num_heads']
        self.hidden_dim = hidden_dim #hparams['hidden_dim']
        self.output_dim = output_dim #hparams['output_dim']
        self.num_layers = num_layers #hparams['num_layers']
        self.dropout_prob = dropout_prob #hparams['dropout_prob']
        self.pred_threshold = pred_threshold #hparams['pred_threshold']
        self.max_visit_num = max_visit_num #hparams['max_visit_num']
        self.code_size = code_size

        # Learning rate parameters
        self.lr = lr #hparams['lr']
        self.wd = wd #hparams['wd']
        self.lr_factor = lr_factor #hparams['lr_factor']
        self.lr_total_iters = lr_total_iters #hparams['lr_total_iters']
        self.mask_prob = 0.2

        # Define node embeddings
        self.emb = nn.Embedding(self.code_size + 1, input_dim) # output_dim = 128 , +1 means CLS token
        self.num_layers = num_layers

        self.cls_emb = torch.nn.Parameter(torch.randn(1, output_dim).to(self.device))
        self.gender_emb = nn.Embedding(5, input_dim)
        self.ethnicity_emb = nn.Embedding(100, input_dim)
                
        self.memory_bank = torch.randn((memory_bank_size, output_dim)).to(self.device)  # Initialize memory bank
        self.memory_bank_size = memory_bank_size


        ####
        # Create a positional encoding matrix (max_len, embed_dim)
        self.position_encoder = PositionalEncoding(d_model=output_dim, 
                                                   dropout=0.2,
                                                   max_len=max_ehr_length + 1,
                                                   max_year=1000)
        
        # Fully connected layer for output
        self.fc = nn.Linear(output_dim, output_dim)
        self.classify = nn.Linear(output_dim, num_class)
        self.num_class = num_class

        self.task = task
    
    def get_mask_subset_with_prob(self, seq_embedding, seq_mask, mask_prob):
        '''
        This function randomly masks a subset of the input sequence embeddings.

        Args:
            seq_embedding (torch.Tensor): Sequence embeddings.
            seq_mask (torch.Tensor): Sequence mask.
            mask_prob (float): Probability of masking.

        Returns:
            masked_embedding (torch.Tensor): Masked sequence embeddings.
            masked_mask (torch.Tensor): Masked sequence mask.
        '''
        seq_mask = ~seq_mask ##real visit is 1, padding is 0
        #print("seq_mask", seq_mask)
        batch, seq_len, device = *seq_mask.shape, seq_mask.device 
        max_masked = math.ceil(mask_prob * seq_len)
        #print(max_masked)

        num_tokens = seq_mask.sum(dim=-1, keepdim=True)
        mask_excess = (seq_mask.cumsum(dim=-1) > (num_tokens * mask_prob).ceil())
        mask_excess = mask_excess[:, :max_masked]

        #print(mask_excess)

        rand = torch.rand((batch, seq_len), device=device).masked_fill(~seq_mask, -1e9)
        _, sampled_indices = rand.topk(max_masked, dim=-1)
        sampled_indices = (sampled_indices + 1).masked_fill(mask_excess, 0)

        new_mask = torch.zeros((batch, seq_len + 1), device=device).bool()
        new_mask.scatter_(1, sampled_indices, True)
        new_mask = new_mask[:, 1:].bool() 

        return seq_embedding, new_mask
    
    
    def forward(self, inputs, pos = None):
        #print(anchor)

        patient_embedding = self.patientEncoder(inputs)
       
        if pos is not None:
            pos_embedding = self.patientEncoder(pos)
        else:
            pos_embedding = None
        
        prob_logits = self.classify(patient_embedding)  ##  [bz, 2]

        return patient_embedding, prob_logits, pos_embedding
    
    def patientEncoder(self, data):
        #print(data.timestamp_within_visits.shape)
        src_emb = data.x ##[bz, max_medical_code, output_dim]
        src_emb = self.emb(src_emb).squeeze()  ##[bz, max_medical_code, output_dim]
        #print("src_emb", src_emb.shape)
        #print("visit_id", data.visit_id.shape)
        cls_emb = self.cls_emb.repeat(src_emb.size(0), 1).unsqueeze(1).to(src_emb.device)  ##[bz, 1, output_dim]
        timestamp_within_visits = data.timestamp_within_visits.gather(1, data.visit_id[:, :, 0].unsqueeze(-1).expand(-1, -1, data.timestamp_within_visits.size(-1)))
        timestamp_between_visits = data.timestamp_between_visits.gather(1, data.visit_id[:, :, 0].unsqueeze(-1).expand(-1, -1, data.timestamp_between_visits.size(-1)))
        #print("timestamp_within_visits", timestamp_within_visits.shape)
        #print("timestamp_within_visits", timestamp_between_visits.shape)
        src_pos_emb = self.position_encoder(src_emb, timestamp_within_visits, timestamp_between_visits)  ##[bz, max_medical_code, output_dim]

        ##if consider gender and ethnicity
        gender_emb = self.gender_emb(data.gender).unsqueeze(1) ##[batch_size, 1, output_dim]
        ethnicity_emb = self.ethnicity_emb(data.ethnicity).unsqueeze(1)  ##[batch_size, 1, output_dim]

        src_emb = torch.concat([cls_emb, gender_emb, ethnicity_emb, src_pos_emb], dim=1)  # Shape: [batch_size, max_medical_code + 3, output_dim]
        src_mask = torch.concat([torch.zeros(src_emb.size(0), 3).bool().to(src_emb.device), data.code_mask.squeeze().bool()], dim=-1)  # Shape: [batch_size, max_medical_code + 3]
        
        # Pass through the Transformer encoder
        src_emb = src_emb.transpose(0, 1)
        for layer in self.model:
            src_emb = layer(src_emb, src_key_padding_mask = src_mask)   
        encoded_output = src_emb

        # Take the mean of the encoder's output for each sequence (for classification)
        encoded_output = encoded_output[0, :, :]  # Shape: [batch_size, output_dim]
        #print("encoded_output", encoded_output)
        
        # Pass through the fully connected layer
        output = self.fc(encoded_output)
        
        return output

    def patientEncoder_old(self, data, src, src_mask=None):

        # Embed the source sequence and add positional encoding
        src_emb = self.position_encoder(src, data.timestamp_within_visit, data.timestamp_between_visit)  # Shape: [batch_size, max_visit_num, output_dim]
        cls_emb = self.cls_emb.repeat(src_emb.size(0), 1).unsqueeze(1).to(src_emb.device)  # Shape: [batch_size, 1, output_dim]
        
        ##if consider gender and ethnicity
        gender_emb = self.gender_emb(data.gender).unsqueeze(1) ##[batch_size, 1, output_dim]
        ethnicity_emb = self.ethnicity_emb(data.ethnicity).unsqueeze(1)  ##[batch_size, 1, output_dim]

        src_emb = torch.concat([cls_emb, gender_emb, ethnicity_emb, src_emb], dim=1)  # Shape: [batch_size, max_visit_num + 3, output_dim]
        src_mask = torch.concat([torch.zeros(src_emb.size(0), 3).bool().to(src_emb.device), src_mask], dim=-1)  # Shape: [batch_size, max_visit_num + 3]
        
        # Pass through the Transformer encoder
        encoded_output = self.model(src = src_emb.transpose(0, 1), src_key_padding_mask = src_mask)  # (seq_len, batch, embed_dim)
        
        # Take the mean of the encoder's output for each sequence (for classification)
        encoded_output = encoded_output[0, :, :]  # Shape: [batch_size, output_dim]
        #print("encoded_output", encoded_output)
        
        # Pass through the fully connected layer
        output = self.fc(encoded_output)
        
        return output

    def generate_src_mask(self, seq_len):
        # Mask for padding tokens (using all ones here, you could customize it based on padding)
        src_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(self.device)
        return src_mask

    def pad_sequence(self, visit_embeddings, batch_first=True):
        ##visit_embeddings: [real_visit_num, output_dim]
        max_visit_num = self.max_visit_num
        padded_embeddings = torch.zeros((max_visit_num, self.output_dim)).to(self.device)
        
        mask = torch.zeros(max_visit_num).to(self.device)
        if visit_embeddings.size(0) < max_visit_num:
            padded_embeddings[:visit_embeddings.size(0), :] = visit_embeddings
            padded_length = max_visit_num - visit_embeddings.size(0)
            mask[-padded_length:] = 1
        else:
            padded_embeddings = visit_embeddings[-max_visit_num:]
        
        return padded_embeddings, mask.bool()
    

    def compute_contrastive_loss(self, anchor_embedding, pos_embedding=None, temperature=0.1):
        # Compute the contrastive loss
        anchor_embedding = F.normalize(anchor_embedding, dim=-1)
        if pos_embedding is not None:
            pos_embedding = F.normalize(pos_embedding, dim=-1)

        # Compute the cosine similarity
        if pos_embedding is not None:
            positive_similarities = torch.sum(anchor_embedding * pos_embedding, dim=-1) / temperature
        else:
            positive_similarities = torch.sum(anchor_embedding * anchor_embedding, dim=-1) / temperature


        mask = ~torch.eye(anchor_embedding.size(0), device=self.device).bool()
        negative_similarities_within_anchor = torch.matmul(anchor_embedding, anchor_embedding.t().to(self.device)) / temperature
        negative_similarities_within_anchor_mask = negative_similarities_within_anchor[mask].view(anchor_embedding.size(0), -1)

        if pos_embedding is not None:
            negative_similarities_with_pos = torch.matmul(anchor_embedding, pos_embedding.t().to(self.device)) / temperature
            negative_similarities_with_pos_mask = negative_similarities_with_pos[mask].view(anchor_embedding.size(0), -1)
            negative_similarities = torch.cat([negative_similarities_within_anchor_mask, negative_similarities_with_pos_mask], dim=-1)
        else:
            negative_similarities = negative_similarities_within_anchor_mask 

        #print("anchor_embeddings is", anchor_embedding.shape)
        if self.memory_bank.shape[0] > 0:
            memory_bank_embeddings = self.memory_bank.clone().detach() 
            negative_similarities_memory_bank = torch.matmul(anchor_embedding, memory_bank_embeddings.t().to(self.device)) / temperature
            negative_similarities = torch.cat([negative_similarities_memory_bank, negative_similarities], dim=-1)

        # Concatenate positive and negative similarities for InfoNCE loss
        logits = torch.cat([positive_similarities.unsqueeze(1), negative_similarities], dim=-1)  # Shape: (batch_size, 1 + num_negatives)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)  # Positives are at index 0

        contrastive_loss = F.cross_entropy(logits, labels)

        return contrastive_loss
    
    # STEP FUNCTION USED FOR TRAINING, VALIDATION, AND TESTING
    def _step(self, input_nodes, pos_graph, neg_graph, subgraph, mode):
        '''Defines the step that is run on each batch of data. PyTorch Lightning handles steps including:
            - Moving data to the correct device.
            - Epoch and batch iteration.
            - optimizer.step(), loss.backward(), optimizer.zero_grad() calls.
            - Calling of model.eval(), enabling/disabling grads during evaluation.
            - Logging of metrics.
        
        Args:
            input_nodes (torch.Tensor): Input nodes.
            pos_graph (dgl.DGLHeteroGraph): Positive graph.
            neg_graph (dgl.DGLHeteroGraph): Negative graph.
            subgraph (dgl.DGLHeteroGraph): Subgraph.
            mode (str): The mode of the step (train, val, test).
        '''

        # Get batch size by summing number of nodes in each node type
        batch_size = sum([x.shape[0] for x in input_nodes.values()])

        # Convert heterogeneous graph to homogeneous graph for efficiency
        # See https://docs.dgl.ai/en/latest/generated/dgl.to_homogeneous.html
        subgraph = dgl.to_homogeneous(subgraph, ndata = ['node_index'])

        node_embeddings = self.forward(subgraph)

        # Set node index of negative graph for decoder
        neg_graph.ndata['node_index'] = pos_graph.ndata['node_index']

        # Compute score from decoder
        pos_scores = self.decoder(subgraph, pos_graph, node_embeddings)
        neg_scores = self.decoder(subgraph, neg_graph, node_embeddings)

        # Compute loss
        loss, metrics, edge_type_metrics = self.compute_loss(pos_scores, neg_scores)

        # Return loss and metrics
        return loss, metrics, edge_type_metrics, batch_size
    

    # TRAINING STEP
    def training_step(self, batch, batch_idx):
        '''Defines the step that is run on each batch of training data.'''

        # Get batch elements
        sample = batch
        patient_embedding, prob_logits, _ = self(sample, None)
        
        import numpy as np
        if sample.label.shape[-1] == 1 or len(sample.label.shape) == 1:
            y_true_one_hot = torch.zeros((sample.label.size(0), self.num_class)).to(self.device)
            y_true_one_hot[torch.arange(sample.label.size(0)).long(), sample.label.long()] = 1
        else:
            y_true_one_hot = sample.label
        #print("sample.label", sample.label)
        #print("prob_logits", prob_logits)
        if self.task == 'lenofstay':
            loss = F.cross_entropy(prob_logits, y_true_one_hot)
        else:
            loss = F.binary_cross_entropy_with_logits(prob_logits, y_true_one_hot)
        
        if self.task == 'lenofstay' or self.task == 'readmission' or self.task == 'mortality':
            prob_logits = F.softmax(prob_logits, dim=-1)
        else:
            prob_logits = torch.sigmoid(prob_logits)
        #entropy_loss = F.binary_cross_entropy_with_logits(prob_logits, y_true_one_hot)
        auc, aupr, f1 = self.compute_metrics(sample.label, prob_logits)

        #self.update_memory_bank(patient_embedding.detach())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size = sample.size(0))
        self.log("train/auc", auc, on_step=True, on_epoch=True, prog_bar=True, batch_size = sample.size(0))
        self.log("train/aupr", aupr, on_step=True, on_epoch=True, prog_bar=True, batch_size = sample.size(0))
        self.log("train/f1", f1, on_step=True, on_epoch=True, prog_bar=True, batch_size = sample.size(0))

        #return entropy_loss, auc, aupr
    
    def compute_metrics(self, label, logits):
        '''Computes the AUROC and AUPR for the given logits and labels.'''
        if self.task == 'lenofstay' or self.task == 'phenotype' or self.task == 'drugrec':
            label_numpy = label.detach().cpu().numpy()
            import numpy as np
            print("label_numpy", label_numpy.size)
            if label_numpy.shape[-1] == 1 or len(label_numpy.shape) == 1:
                y_true_one_hot = np.zeros((label_numpy.size, self.num_class))
                y_true_one_hot[np.arange(label_numpy.size), label_numpy.flatten().astype(int)] = 1
            else:
                y_true_one_hot = label_numpy

            logits = logits.detach().cpu().numpy()
           
            auroc = roc_auc_score(y_true_one_hot, logits, average = 'micro')
            aupr = average_precision_score(y_true_one_hot, logits, average = 'micro')
            f1 = f1_score(y_true_one_hot, (logits >= 0.2).astype(int), average = 'weighted')
            return auroc, aupr, f1
        else:
            logits = logits[:, 1].detach().cpu().numpy()
            if torch.sum(label) == 0:
                auc = 0.5
                aupr = 0.5
            else:
                auc = roc_auc_score(label.cpu().numpy(), logits)
                aupr = average_precision_score(label.cpu().numpy(), logits)
            logits_i_int = [1 if j > 0.5 else 0 for j in logits]
            f1 = f1_score(label.cpu().numpy(), logits_i_int)
            return auc, aupr, f1
    
    def update_memory_bank(self, new_embeddings):     
        # Update the memory bank with new embeddings
        tensors_gatherd = [torch.zeros_like(new_embeddings) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gatherd, new_embeddings)

        new_embeddings = F.normalize(torch.cat(tensors_gatherd, dim=0), dim=-1)
        self.memory_bank = torch.cat([self.memory_bank, new_embeddings.to(self.memory_bank.device)], dim=0)

        # Limit the size of the memory bank
        if self.memory_bank.shape[0] > self.memory_bank_size:
            self.memory_bank = self.memory_bank[-self.memory_bank_size:]    


    # VALIDATION STEP
    def validation_step(self, batch, batch_idx):
        '''Defines the step that is run on each batch of validation data.'''

        # Get batch elements
        sample = batch
        batch_size = sample.size(0)

        # Get patient embeddings
        patient_embedding, prob_logits, _ = self(sample, None)

        # Compute loss
        #loss = F.cross_entropy(prob_logits, sample.label)
        import numpy as np
        if sample.label.shape[-1] == 1 or len(sample.label.shape) == 1:
            y_true_one_hot = torch.zeros((sample.label.size(0), self.num_class)).to(self.device)
            y_true_one_hot[torch.arange(sample.label.size(0)).long(), sample.label.long()] = 1
        else:
            y_true_one_hot = sample.label
        #print("sample.label", sample.label)
        #print("prob_logits", prob_logits)
        if self.task == 'lenofstay':
            loss = F.cross_entropy(prob_logits, y_true_one_hot)
        else:
            loss = F.binary_cross_entropy_with_logits(prob_logits, y_true_one_hot)

        print("logits")
        print(prob_logits.shape)
        # Compute metrics
        if self.task == 'lenofstay' or self.task == 'readmission' or self.task == 'mortality':
            prob_logits = F.softmax(prob_logits, dim=-1)
        else:
            prob_logits = torch.sigmoid(prob_logits)

        auc, aupr, f1 = self.compute_metrics(sample.label, prob_logits)

        # Record loss and metrics
        values = {"val/loss": loss.detach(),
                  "val/auc": auc,
                  "val/aupr": aupr,
                  "val/f1": f1
                  }
        self.log_dict(values, batch_size = batch_size)


    # TEST STEP
    def test_step(self, batch, batch_idx):
        '''Defines the step that is run on each batch of test data.'''

        # Get batch elements
        sample = batch
        batch_size = sample.size(0)

        # Get patient embeddings
        patient_embedding, prob_logits, _ = self(sample, None)

        import numpy as np
        if sample.label.shape[-1] == 1 or len(sample.label.shape) == 1:
            y_true_one_hot = torch.zeros((sample.label.size(0), self.num_class)).to(self.device)
            y_true_one_hot[torch.arange(sample.label.size(0)).long(), sample.label.long()] = 1
        else:
            y_true_one_hot = sample.label
        #print("sample.label", sample.label)
        #print("prob_logits", prob_logits)
        if self.task == 'lenofstay':
            loss = F.cross_entropy(prob_logits, y_true_one_hot)
        else:
            loss = F.binary_cross_entropy_with_logits(prob_logits, y_true_one_hot)

        # Compute metrics
        if self.task == 'lenofstay' or self.task == 'readmission' or self.task == 'mortality':
            prob_logits = F.softmax(prob_logits, dim=-1)
        else:
            prob_logits = torch.sigmoid(prob_logits)

        auc, aupr, f1 = self.compute_metrics(sample.label, prob_logits)

        # Record loss and metrics
        values = {"test/loss": loss.detach(),
                  "test/auc": auc,
                  "test/aupr": aupr,
                  "test/f1": f1
                  }
        self.log_dict(values, batch_size = batch_size)
    
    def predict_step(self, batch, batch_idx):
        '''Defines the step that is run on each batch of test data.'''

        # Get batch elements
        patient_subgraph, _ = batch

        # Get patient embeddings
        patient_embeddings, prob_logits, _ = self(patient_subgraph)
        prob_true = prob_logits[:, 1]
        return patient_embeddings, prob_true
        
    # LOSS FUNCTION
    def compute_loss(self, pos_scores, neg_scores):
        pass
    

    # OPTIMIZER AND SCHEDULER
    def configure_optimizers(self):
        '''
        This function is called by PyTorch Lightning to get the optimizer and scheduler.
        We reduce the learning rate by a factor of lr_factor if the validation loss does not improve for lr_patience epochs.

        Returns:
            dict: Dictionary containing the optimizer and scheduler.
        '''
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)
        
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    'name': 'curr_lr',
                    },
                }


    # QUERY MODEL FOR EMBEDDINGS
    @torch.no_grad()
    def get_embeddings(self, dataloader, device):
        self.to(device)
        self.eval()
        embeddings = []
        for idx, batch in tqdm(dataloader, desc = 'Getting embeddings'):
            print(batch)
            x = batch
            x = x.to(device)
            embedding, _ = self(x)
            print("get_embeddings", embedding)
            embeddings.append(embedding)
        return torch.cat(embeddings, dim=0)


    # QUERY MODEL FOR SCORES
    @torch.no_grad()
    def get_scores(self, src_indices, dst_indices, query_edge_type, hparams, query_kg = None,
                   use_cache = True, degree_threshold = None, fixed_k = None):
        '''
        For a set of edges described by paired source nodes and destination nodes, this function 
        computes the likelihood score of each edge. Note that `src_indices` must be valid global
        node IDs of type `query_edge_type[0]` (where the global node IDs are stored as a node
        attribute in `query_kg.ndata['node_index']`, and `dst_indices` must be valid global node
        IDs of type `query_edge_type[2]`.

        Args:
            src_indices (list): List of global (not reindexed!) source node indices.
            dst_indices (list): List of global (not reindexed!) destination node indices.
            query_edge_type (tuple): Edge type of query edges.
            hparams (dict): Dictionary of model hyperparameters.
            query_kg (dgl.DGLHeteroGraph): Query graph.
            use_cache (bool): See subsample_graph().
            degree_threshold (int): See subsample_graph().
            fixed_k (int): See subsample_graph().
        '''

        # Subsample graph
        query_kg, query_sampler = self.subsample_graph(query_kg, hparams, use_cache, degree_threshold, fixed_k)

        # Define query nodes
        query_indices = torch.tensor(src_indices + dst_indices) # .to(device)
        query_indices = torch.unique(query_indices).unsqueeze(1)

        # Define query edge type
        if query_edge_type not in query_kg.canonical_etypes:
            raise ValueError('Edge type not in knowledge graph.')
        src_type = query_edge_type[0]
        dst_type = query_edge_type[2]

        # Get reindexed node indices
        kg_indices = query_kg.ndata['node_index']
        query_nodes = {key: torch.where(value == query_indices)[1] for key, value in kg_indices.items()}

        # If source and destination types are not the same
        if src_type != dst_type:

            # Map source indices to reindexed values
            src_set = list(set(src_indices))
            src_srt = list(range(len(src_set)))
            src_map = {x: y for x, y in zip(src_set, src_srt)}
            src_nodes = torch.tensor([src_map[x] for x in src_indices])

            # Map destination indices to reindexed values
            dst_set = list(set(dst_indices))
            dst_srt = list(range(len(dst_set)))
            dst_map = {x: y for x, y in zip(dst_set, dst_srt)}
            dst_nodes = torch.tensor([dst_map[x] for x in dst_indices])

        # If source and destination types are the same
        else:

            # Map source and destination indices to reindexed values
            src_dst_set = list(set(src_indices + dst_indices))
            src_dst_srt = list(range(len(src_dst_set)))
            src_dst_map = {x: y for x, y in zip(src_dst_set, src_dst_srt)}
            src_nodes = torch.tensor([src_dst_map[x] for x in src_indices])
            dst_nodes = torch.tensor([src_dst_map[x] for x in dst_indices])

        # Make query graph
        edge_graph_data = {etype: ([], []) for etype in query_kg.canonical_etypes}
        edge_graph_data[query_edge_type] = (src_nodes, dst_nodes)
        query_edge_graph = dgl.heterograph(edge_graph_data)
        assert(query_kg.ntypes == query_edge_graph.ntypes)
        assert(query_kg.canonical_etypes == query_edge_graph.canonical_etypes)

        # If source and destination types are not the same
        if src_type != dst_type:

            # Get global node indices
            src_map_rev = {y: x for x, y in src_map.items()}
            dst_map_rev = {y: x for x, y in dst_map.items()}
            global_src_nodes = torch.tensor([src_map_rev[x.item()] for x in query_edge_graph.nodes(src_type)])
            global_dst_nodes = torch.tensor([dst_map_rev[x.item()] for x in query_edge_graph.nodes(dst_type)])

            # Add global node indices
            node_index_data = {ntype: torch.empty(0) for ntype in query_kg.ntypes}
            node_index_data[src_type] = global_src_nodes
            node_index_data[dst_type] = global_dst_nodes
            query_edge_graph.ndata['node_index'] = node_index_data

        # If source and destination types are the same
        else:

            # Get global node indices
            src_dst_map_rev = {y: x for x, y in src_dst_map.items()}
            global_src_dst_nodes = torch.tensor([src_dst_map_rev[x.item()] for x in query_edge_graph.nodes(src_type)])

            # Add global node indices
            node_index_data = {ntype: torch.empty(0) for ntype in query_kg.ntypes}
            node_index_data[src_type] = global_src_dst_nodes
            query_edge_graph.ndata['node_index'] = node_index_data

        # Get subgraph
        _, _, query_subgraph = query_sampler.sample(query_kg, query_nodes)

        # Convert heterogeneous graph to homogeneous graph for efficiency
        query_subgraph = dgl.to_homogeneous(query_subgraph, ndata = ['node_index'])

        # # Send to device
        # query_subgraph = query_subgraph.to(device)
        # query_edge_graph = query_edge_graph.to(device)

        # Get node embeddings
        node_embeddings = self.forward(query_subgraph)

        # Get scores
        scores = self.decoder(query_subgraph, query_edge_graph, node_embeddings)
        # scores = torch.sigmoid(scores[query_edge_type]) # apply sigmoid with BCE loss
        scores = torch.sigmoid(scores[query_edge_type])

        return scores

    
    # QUERY MODEL FOR SCORES
    @torch.no_grad()
    def get_scores_from_embeddings(self, src_indices, dst_indices, query_edge_type, hparams, 
                                   query_kg = None, use_cache = True, embeddings = None):
        '''
        For a set of edges described by paired source nodes and destination nodes, this function 
        computes the likelihood score of each edge. Note that `src_indices` must be valid global
        node IDs of type `query_edge_type[0]` (where the global node IDs are stored as a node
        attribute in `query_kg.ndata['node_index']`, and `dst_indices` must be valid global node
        IDs of type `query_edge_type[2]`.

        This function differs from get_scores() in that it uses cached embeddings, and is therefore much faster.

        Args:
            src_indices (list): List of global (not reindexed!) source node indices.
            dst_indices (list): List of global (not reindexed!) destination node indices.
            query_edge_type (tuple): Edge type of query edges.
            hparams (dict): Dictionary of model hyperparameters.
            query_kg (dgl.DGLHeteroGraph): Query graph.
            embeddings (torch.Tensor): Node embeddings saved by save_embeddings() in pretrain.py. If not provided, the 
                embeddings are read from disk based on the values of hparams['save_dir'] and hparams['best_ckpt'].
        '''

        # If graph is not cached, raise error
        if query_kg is None and not use_cache:
            raise ValueError('Either query_kg must be provided or use_cache must be True.')
        elif use_cache:
            # Check if cached knowledge graph exists
            if self.cached_kg is None:
                raise ValueError('Cached knowledge graph does not exist. Call cache_graph() first.')
            # Use cached knowledge graph
            query_kg = self.cached_kg

        # Load embeddings if not provided
        if embeddings is None:
            embed_path = hparams['save_dir'] / 'embeddings' / (hparams['best_ckpt'].split('.ckpt')[0] + '_embeddings.pt')
            embeddings = torch.load(embed_path)
        
        # Retrieve cached embeddings
        src_embeddings = embeddings[src_indices]
        dst_embeddings = embeddings[dst_indices]

        # Apply activation function
        src_embeddings = F.leaky_relu(src_embeddings)
        dst_embeddings = F.leaky_relu(dst_embeddings)

        # Get relation weight for specific edge type
        edge_type_index = [i for i, etype in enumerate(query_kg.canonical_etypes) if etype == query_edge_type]
        if len(edge_type_index) == 0:
            raise ValueError(f"Edge type ({query_edge_type[0]}, {query_edge_type[1]}, {query_edge_type[2]}) not found in knowledge graph.")
        else:
            edge_type_index = edge_type_index[0]
        rel_weights = self.decoder.relation_weights[edge_type_index]

        # Compute weighted dot product
        scores = torch.sum(src_embeddings * rel_weights * dst_embeddings, dim = 1)
        scores = torch.sigmoid(scores)
        return scores