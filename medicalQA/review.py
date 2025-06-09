import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple

from transformers import LlamaForCausalLM
import numpy as np
import torch.nn.functional as F

#pretrained_emb_path = '/n/holylfs06/LABS/mzitnik_lab/Lab/shgao/timetok/TOTEM/forecasting/forecasting_tokenizer_embeddings/MCQ_1_TS/train/padded_x_codes_all.npy'
#pretrained_emb_mask_path = '/n/holylfs06/LABS/mzitnik_lab/Lab/shgao/timetok/TOTEM/forecasting/forecasting_tokenizer_embeddings/MCQ_1_TS/train/padded_masks.npy'
class Review(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        num_prefix: int,
        pad_id: int,
        hidden_dim: int,
        #kge_model: str,
        pretrain_emb_path: str,
        pretrained_emb_mask_path: str,
    ) -> None:
        super(Review, self).__init__()
        self.llama_model = model
        self.max_length = 2048
        ##for 8B 4096, for qwen 3584
        self.llm_dim = self.llama_model.config.hidden_size  # 4096 for Llama-8B, 3584 for Qwen-7B
        self.pad_id = pad_id
        self.device = self.llama_model.device
        self.projector = nn.Linear(64, self.llm_dim)
        
        print("Adapter Load From {}".format(pretrain_emb_path))
        embeddings = np.load(pretrain_emb_path).squeeze()
        print(embeddings.shape)

        padded_mask = np.load(pretrained_emb_mask_path).squeeze()
        print(padded_mask.shape)
        #padded_mask = np.load(pretrained_emb_mask_path).squeeze()
        
        #miss_emb = torch.nn.Parameter(torch.randn(10000, embeddings.shape[-1]), requires_grad=False)
        #embeddings = np.concatenate((embeddings, miss_emb), axis=0)
        self.embeddings = torch.tensor(embeddings).to(self.llama_model.device)  # Move embeddings to the same device as the model
        self.embeddings.to(self.device)

        self.embeddings_pad = torch.tensor(padded_mask).to(self.llama_model.device)  # Move embeddings to the same device as the model
        self.embeddings_pad.to(self.device)
        if torch.isnan(self.embeddings).any():
            print("NaN detected in inputs!")
    
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
        
        attention_mask_new = torch.zeros((bz, 512+2048)).cuda()
        concat_embeds = torch.ones((bz, 512+2048, self.llm_dim)).cuda() * self.llama_model.model.model.embed_tokens(torch.tensor(self.pad_id).cuda())
        new_labels = torch.full((bz, 512+2048), fill_value=-100, dtype=torch.long).cuda()  # Initialize new_labels with -100 for padding
        #new_labels = torch.cat((new_labels), dim=1)  # pad the labels to match the new input size
        for i in range(bz):
            #print("input_ids[i]", input_ids[i])
            #question_id = input_ids[i][0]
            #input_id_i = input_ids[i][1:]

            attention_mask_i = attention_mask[i]
            first_non_pad_idx_i = torch.argmax(attention_mask_i).item()
            #print("first_non_pad_idx_i", first_non_pad_idx_i)

            question_id = input_ids[i][first_non_pad_idx_i]
            input_id_i = input_ids[i][first_non_pad_idx_i+1:]

            #print("question_id", question_id)
            ts_embeddings = self.embeddings[question_id]
            #print("ts_embeddings", ts_embeddings.shape)
            ts_embeddings_mask = torch.sum(self.embeddings_pad[question_id]).item()
            #print("ts_embeddings_mask", ts_embeddings_mask)
            if ts_embeddings_mask > 512:
                ts_embeddings_mask = 512
            ts_embeddings = self.projector(ts_embeddings[:int(ts_embeddings_mask)]) ##
            
        
            query_embeds = self.llama_model.model.model.embed_tokens(input_id_i)
            input_embeds_i = torch.cat((ts_embeddings, query_embeds), dim=0)
            #print(input_embeds_i.shape)
            concat_embeds[i, -input_embeds_i.shape[0]:] = input_embeds_i
            #print("concat_embeds", concat_embeds.shape)
            attention_mask_new[i, -input_embeds_i.shape[0]:] = 1
            new_labels[i, -query_embeds.shape[0]:] = labels[i, -query_embeds.shape[0]:]  # Assign the labels for the query part
        

        output = self.llama_model(
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

        #print("output",output)
        return output