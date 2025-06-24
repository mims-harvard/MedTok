import torch
import os
import argparse
import torch
from vllm import LLM
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, set_seed
import re
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import os
import logging
from openai import AzureOpenAI
from tqdm import tqdm
from transformers import set_seed
import re
import pickle
import json

# Build your client
client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"), # Obtained from the team's key manager
  api_version="2024-05-01-preview"
)


def llm_output(query, new_tokens_num):

    messages = [
        {"role": "system", "content": "You are a medical coding assistant."},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model='gpt-4-turbo', # Model deployment name      
        max_tokens = new_tokens_num,
        messages=messages
    )
    
    return response.choices[0].message.content

class AfrimedLoader:
    def __init__(self, data='mcq_expert', dir="../Dataset/MedicalQA/", **kwargs):
        print("data is {}".format(data))
        if data == 'AfrimedQA-MCQ':
            self.data = 'mcq_expert'
        elif data == 'AfrimedQA-SAQ':
            self.data = 'saq_expert' 
        
        benchmark = self.process_dataset(dir)
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        self.dataset = benchmark[self.data]
        print("{} has {} queries".format(data, len(self.dataset)))
        print(self.dataset)
        self.index = sorted(self.dataset.keys())
    
    def process_dataset(self, dir):       
        
        dataset_name = self.data
        datafile_name = "AfrimedQA_{}.json".format(dataset_name)

        print(datafile_name)

        if os.path.exists(os.path.join(dir, datafile_name)):
            dataset = json.load(open(os.path.join(dir, datafile_name)))
            return dataset
        else:
            from datasets import load_dataset
            options = [" A: ", " B: ", " C: ", " D: ", " E: ", " F: "]
            ds = load_dataset("intronhealth/afrimedqa_v2")['train']
            dataset = {dataset_name: {}}
            print("dataset is {}".format(dataset_name))
            
            for d in ds:
                print(d['question_type'])
                #if d['split'] == 'train':
                #    continue
                if d['tier'] != 'expert':
                    continue
                if d['question_type'] == 'mcq' and 'mcq' in dataset_name:
                    choices = [v for k, v in json.loads(d["answer_options"]).items()]
                
                    text = d['question_clean'].strip() + "\n"
                    for j in range(len(choices)):
                        text += "{} {}\n".format(options[j], choices[j])

                    label_index = int(d['correct_answer'][6])-1
                    answer = chr(ord('A') + label_index)
                    answer_content = choices[label_index]

                    idx = len(dataset['mcq_expert'])
                    dataset['mcq_expert'][idx] = {"query": text, "answer": answer, "answer_index": label_index, "answer_content": answer_content}
                if d['question_type'] == 'saq' and 'saq' in dataset_name:
                    
                    text = d['question_clean'].strip() + "\n"
                    answer = d['answer_rationale'].strip().replace('\n', ' ').replace('\r', '')

                    #label_index = int(d['correct_answer'][6])-1
                    #answer = chr(ord('A') + label_index)
                    #answer_content = choices[label_index]

                    idx = len(dataset['saq_expert'])
                    dataset['saq_expert'][idx] = {"query": text, "answer": answer, "answer_index": None, "answer_content": None}

            with open(os.path.join(dir, datafile_name), 'w') as f:
                json.dump(dataset, f, indent=2)

        return dataset

    def __process_data__(self, key):
        data = self.dataset[self.index[key]]

        answer = data["answer"].strip()
        if self.data == 'saq_expert':
            label_index = answer
        else:        
            label_index = ord(answer) - ord('A')
        
        return {"text": data['query'], "answer": answer, "answer_index": label_index}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.__process_data__(key)
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")

import json
import openai

# Example function to get structured JSON response from GPT
def get_icd_codes(query):
    prompt = f"""
    Please think deeply. Identify all medical entities from the given input query and return their corresponding medical codes (ICD-10, RxNorm, LOINC, SNOMED CT, ATC).
    
    Input:
    "{query}"
    
    Output format (JSON):
    {{
        "Diseases": [
            {{"Disease": "Example Disease", "ICD-10": "X00.0", "ICD-9": "000.0"}}
        ],
        "Medications":[
            {{"Medication": "Example Medication", "ATC": "X00", "NDC": "0000000000"}}
        ],
        "Procedures": [
            {{"Procedure": "Example Procedure", "ICD-10-PCS": "X0000", "ICD-9-CM": "0000"}}
        ]
    }}
    """

    try:
        response = llm_output(prompt, new_tokens_num=256)
        response_text = response
        print(response_text)
    except Exception as e:
        return None
    
    # Parse JSON response
    try:
        icd_data = json.loads(response_text)
        return icd_data
    except KeyError:
        return {"error": "KeyError: The response does not contain the expected keys."}
    except TypeError:
        return {"error": "TypeError: The response is not in the expected format."}
    except json.JSONDecodeError:
        try:
            response = llm_output(prompt, new_tokens_num=512)  # Retry with more tokens
            response_text = response
        except Exception as e:
            return {"error": f"Error during retry: {str(e)}"}
        try:
            icd_data = json.loads(response_text)
            return icd_data
        except Exception as e:
            return {"error": "JSONDecodeError: The response is not valid JSON."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


def main(args):

    ##load llm
    set_seed(42)

    logging.basicConfig(filename = args.dataset + ".log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    flag = 0
    code_dict_for_each_query = {}  
    ##load data
    if args.dataset in ['Afrimedqa']:
        data = AfrimedLoader(data='AfrimedQA-MCQ')  # Load the MCQ dataset
    
    for idx, d in tqdm(enumerate(data),  total=len(data), desc=f'Evaluating data'):
        if 'text' in d:
            query = d['text']
        elif 'query' in d: 
            query = d['query']
        elif 'question' in d:
            query = d['question']

        query_stem = query.split('\n')[0]
        print(query_stem)
        icd_codes = get_icd_codes(query_stem)
        print(icd_codes)
        
        if icd_codes == None:
            code_dict_for_each_query[idx] = {'icd10': None, 'icd9': None}
            flag += 1
        elif 'error' in icd_codes:
            code_dict_for_each_query[idx] = {'icd10': None, 'icd9': None}
            flag += 1
        else:
            code_dict_for_each_query[idx] = {}
            # Extract ICD codes for diseases
            for k, v in icd_codes.items(): #v is a list
                for v_i in v:
                    for k_i, v_i_k in v_i.items():
                        if k_i == 'Disease' or k_i == 'Medication' or k_i == 'Procedure':
                            continue
                        elif v_i_k is None or v_i_k == "N/A" or v_i_k == "":
                            continue
                        else:
                            if k_i not in code_dict_for_each_query[idx]:
                                code_dict_for_each_query[idx][k_i] = []
                            if isinstance(v_i_k, list):
                                for code in v_i_k:
                                    if code not in code_dict_for_each_query[idx][k_i]:
                                        code_dict_for_each_query[idx][k_i].append(code)
                            else:
                                code_dict_for_each_query[idx][k_i].append(v_i_k)
            print(code_dict_for_each_query[idx])
        
    #print(flag)
    #print(code_dict_for_each_query)
                
    with open('query_icd_codes_' + args.dataset + '.json', 'w') as f:
        json.dump(code_dict_for_each_query, f)

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='Afrimedqa', type=str)
    parser.add_argument("--key", type=str)

    args = parser.parse_args()
    main(args)


