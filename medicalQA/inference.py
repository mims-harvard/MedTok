import os
import json
import torch
import transformers
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from utils.prompter import Prompter
import numpy as np
import torch.nn.functional as F
prompter = Prompter("alpaca")

base_path = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
cutoff_len = 256

def load_test_dataset(path):
    test_dataset = json.load(open(path, "r"))
    return test_dataset


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 256
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point, inference=True):
    ##give an example

    query, output = data_point['input'][:2]
    medical_tokens = data_point['medical_codes']

    instruction = "The following is a multiple-choice medical question. Please directly select and provide the correct answer from options 'A', 'B, 'C', 'D'. Only return the correct answer by 'A', 'B', 'C', or 'D'."

    if inference:
        # If inference is True, we want to generate a prompt without an answer
        full_prompt = prompter.generate_prompt(
            instruction,
            "\nThe question is: \n{query}\n".format(query=query),
            ""
        )
        # No output provided, just the question
        # This is useful for inference where we want to generate an answer
    else:
        full_prompt = prompter.generate_prompt(
            instruction,
            "\nThe question is: \n{query}\n Answer: ".format(query=query),
            output,
        )
    print("Full Prompt:", full_prompt)  # Debugging line to see the full prompt
    tokenized_full_prompt = tokenize(full_prompt)

    medical_tokens_max_length = [0 for _ in range(cutoff_len)]
    medical_tokens_max_length[:len(medical_tokens)] = medical_tokens
    medical_tokens_attention_mask = [0 for _ in range(cutoff_len)]
    medical_tokens_attention_mask[:len(medical_tokens)] = [1 for _ in range(len(medical_tokens))]
    medical_tokens_label = [-100 for _ in range(cutoff_len)]  # Initialize with -100 for padding
    if len(medical_tokens) > 0:
        medical_tokens_label[:len(medical_tokens)] = medical_tokens

    tokenized_full_prompt['input_ids'] = medical_tokens_max_length + tokenized_full_prompt['input_ids']
    tokenized_full_prompt['attention_mask'] = medical_tokens_attention_mask + tokenized_full_prompt['attention_mask']
    tokenized_full_prompt['labels'] = medical_tokens_label + tokenized_full_prompt['labels']
        
    
    return tokenized_full_prompt


if __name__ == "__main__":
    cuda = "cuda:0"
    lora_weights = "r8_alpha_16_bz_256_epoch_1_llama3.1_lr_0.00001_review_ratio_0.8/"
    test_data_path = "mmlu_dataset.json"
    #embedding_path = "primekg/embeddings.pth"
    pretrain_emb_path: str = 'MedTok/embeddings_all_3000.npy'
    embeddings = np.load(pretrain_emb_path)
    miss_emb = torch.nn.Parameter(torch.randn(100, embeddings.shape[-1]), requires_grad=False)
    embeddings = np.concatenate((embeddings, miss_emb), axis=0)
    embeddings = torch.tensor(embeddings).cuda()

        
    test_dataset = load_dataset("json", data_files=test_data_path)
    #test_dataset = load_dataset(test_data_path)
    #kg_embeddings = torch.load(embedding_path, map_location='cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    print("tokenizers")
    
    projector = torch.load("r8_alpha_16_bz_256_epoch_1_llama3.1_lr_0.00001_review_ratio_0.8/projector.pth", map_location='cuda:0')
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        #torch_dtype=torch.bfloat16
    )#.to(cuda)
    #model.load_adapter(lora_weights, use_safetensors=True)
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        #use_safetensors=True
        #torch_dtype=torch.bfloat16,
    ).to(cuda)
    model.config.pad_token_id = tokenizer.eos_token_id 
    model = model.eval()
   
    test_dataset = test_dataset.map(generate_and_tokenize_prompt)
    test_dataset = test_dataset['train']
    results = []
    for _, data in tqdm(enumerate(test_dataset), total=len(test_dataset), desc=f'Evaluating data'):
        label = data['input'][-1]
        input_id_i = torch.tensor(data['input_ids']).cuda()
        attention_mask_i = torch.tensor(data['attention_mask']).cuda()
        bz = 1
        #print(input_ids.shape)
        attention_mask_new = torch.zeros((1, 512)).cuda()
        concat_embeds = torch.zeros((512, 4096)).cuda() * model.model.model.embed_tokens(torch.tensor(tokenizer.eos_token_id).cuda())
        new_labels = torch.full((1, 512), fill_value=-100, dtype=torch.long).cuda()
        
        first_non_pad_idx_i = torch.argmax(attention_mask_i).item()
        medical_tokens_i = torch.tensor(input_id_i[first_non_pad_idx_i:256])
        attention_mask_medical_tokens_i = attention_mask_i[first_non_pad_idx_i:256]
        medical_tokens_i = medical_tokens_i[attention_mask_medical_tokens_i == 1]


        medical_embeddings = embeddings[medical_tokens_i.long()]
        medical_embeddings = projector(medical_embeddings) 
        #medical_embeddings = torch.mean(medical_embeddings, dim=0).unsqueeze(0)

        query_embeds = model.model.model.embed_tokens(input_id_i[256:])
        input_embeds_i = torch.cat((medical_embeddings, query_embeds), dim=0)
            
        
        concat_embeds[-input_embeds_i.shape[0]:] = input_embeds_i
        attention_mask_new[:, -input_embeds_i.shape[0]:] = 1
        
    
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        try:
            generated_ids = model.generate(
                attention_mask=attention_mask_new,
                inputs_embeds=concat_embeds.unsqueeze(0),
                max_new_tokens=64,
                do_sample = True,
                temperature=0.4,  # Adjust temperature for sampling
                top_p=0.9,  # Adjust top-p for sampling
                eos_token_id = terminators
            )
        except Exception as e:
            print(f"Error during generation: {e}")
            print("Skipping this instance due to generation error.")
            results.append(0)  # Append 0 to indicate failure for this instance
            continue  # Skip to the next instance

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().replace('\n', ' ').replace('\'', '').replace('(', '').replace(')', '')
        print("response", response)
        if len(response) == 0:
            print("Empty generated text")
            results.append(0)
            continue
        if label in response[1:]:
            predict_answer = label
        elif "Answer: " in response:
            answer_index = response.find("Answer: ")
            predict_answer = response[answer_index + 8:].strip()[0]  # Take the first character after "Answer: "
        elif "The correct answer is " in response:
            answer_index = response.find("The correct answer is ")
            predict_answer = response[answer_index+22:].strip()[0]  
        elif "The correct answer is:" in response:
            answer_index = response.find("The correct answer is:")
            predict_answer = response[answer_index+len("The correct answer is:"):].strip()[0]  
        elif "the correct answer is " in response:
            answer_index = response.find("the correct answer is ")
            predict_answer = response[answer_index+21:].strip()[0] 
            answer_index = response.find("The answer is ")
            predict_answer = response[answer_index+14:].strip()[0]  
            answer_index = response.find("The answer is: ")
            predict_answer = response[answer_index+15:].strip()[0]  
        elif "the answer is " in response:
            answer_index = response.find("the answer is ")
            predict_answer = response[answer_index+14:].strip()[0] 
        elif "the answer is: " in response:
            answer_index = response.find("the answer is: ")
            predict_answer = response[answer_index+15:].strip()[0] 
            answer_index = response.find("assistant ")
            predict_answer = response[answer_index+len("assistant "):].strip()[0]  
            if len(response) > 0:
                predict_answer = response[0]
            else:
                predict_answer = 'None'
        elif "assistant " in response:
            answer_index = response.find("assistant")
            predict_answer = response[answer_index+9:].strip()[0] 
        else:
            predict_answer = response[0]
        print("predicted answer", predict_answer)

        
        if predict_answer == label:
            results.append(1)
        else:
            results.append(0)

    print("Accuracy: ", sum(results)/len(results))
       
