import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple

from transformers import LlamaForCausalLM
import numpy as np
import torch.nn.functional as F

class Review(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        num_prefix: int,
        hidden_dim: int,
        kge_model: str,
        pretrain_emb_path: str = '../MedTok/embeddings_all.npy'
    ) -> None:
        super(Review, self).__init__()
        self.llama_model = model
        self.max_length = 256
        ##for 8B 4096

        self.device = self.llama_model.device
        self.projector = nn.Linear(256, 4096)
        
        print("Adapter Load From {}".format(pretrain_emb_path))
        embeddings = np.load(pretrain_emb_path)
        self.embeddings = torch.tensor(embeddings).cuda()
        self.embeddings.to(self.device)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        embedding_ids: torch.LongTensor = None
    ):  
       
        bz = input_ids.shape[0]
        attention_mask_new = torch.zeros((bz, 512)).cuda()
        concat_embeds = torch.ones((bz, 512, 4096)).cuda() * self.llama_model.model.model.embed_tokens(torch.tensor(128009).cuda())
        new_labels = torch.full((bz, 512), fill_value=-100, dtype=torch.long)
        for i in range(bz):
            input_id_i = input_ids[i]
            attention_mask_i = attention_mask[i]  
            

            first_non_pad_idx_i = torch.argmax(attention_mask_i).item()
            medical_tokens_i = input_id_i[first_non_pad_idx_i:self.max_length]
            attention_mask_medical_tokens_i = attention_mask_i[first_non_pad_idx_i:self.max_length]
            medical_tokens_i = medical_tokens_i[attention_mask_medical_tokens_i == 1]
            

            medical_embeddings = self.embeddings[medical_tokens_i.long()]
            medical_embeddings = F.normalize(medical_embeddings, p=2, dim=-1)
            medical_embeddings = torch.mean(medical_embeddings, dim=0).unsqueeze(0)
            medical_embeddings = self.projector(medical_embeddings) 


            query_embeds = self.llama_model.model.model.embed_tokens(input_id_i[self.max_length:])
            input_embeds_i = torch.cat((medical_embeddings, query_embeds), dim=0)
            

            concat_embeds[i, -input_embeds_i.shape[0]:] = input_embeds_i
            attention_mask_new[i, -input_embeds_i.shape[0]:] = 1
            
            new_labels[i, -query_embeds.shape[0]:] = labels[i, :]

        
        return self.llama_model(
            input_ids=None,
            attention_mask=attention_mask_new,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=concat_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )