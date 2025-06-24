import numpy as np
import json
import pandas as pd
import os

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


def read_medical_code():
    med_codes_pkg_map_path = '../Dataset/medicalCode/all_codes_mappings_v3.parquet'
    medical_code = pd.read_parquet(med_codes_pkg_map_path)
    medical_code['med_code'] = medical_code['med_code'].apply(lambda x: x.replace('.', ''))
    #print(self.medical_code.head())
    medical_code_range = {}
    for idx, m in enumerate(medical_code['med_code']):
        if '-' in m and '.' in m:
            medical_code_range[m] = idx
    return medical_code, medical_code_range


def read_query_icd_codes(dataset):
    with open(f'query_icd_codes_{dataset}.json'.format(dataset), 'r') as f:
        code_dict_for_each_query = json.load(f)
    return code_dict_for_each_query

def is_in_general_range(value, range_string):
    import re
    #print(value, range_string)
    left, right = range_string.split('-')[:2]
    #print(left, right)
    if value >= left and value <= right:
        return True
    else:
        return False
    
dataset = 'Afrimedqa'

medical_code_vocabulary, medical_code_range = read_medical_code()
code_dict_for_each_query = read_query_icd_codes(dataset)
if dataset in ['Afrimedqa']:
    data = AfrimedLoader("AfrimedQA-MCQ")

code_mapped = {}
code_mapped_query = {}

difficult = 0
middle = 0
easy = 0
middle_idx = []
easy_idx = []
difficult_idx = []
for idx, d in enumerate(data):
    print("Processing query index:", idx)

    code_d = code_dict_for_each_query[str(idx)]
    print(code_d)
    
    code_mapped_query[idx] = []
    #"4": {"SNOMED CT": ["113197003"]}

    if len(code_d) == 0:
        code_mapped_query[idx] = [len(medical_code_vocabulary)]  ##denote the null
        difficult += 1
    else:
        for k, v in code_d.items():
            if k == 'ICD-9' or k == 'ICD-10':
                if v is None:
                    continue
                elif len(v) == 0:
                    continue
                else:
                    for c in v:
                        c = c.replace('.', '')  # Remove any periods from the code
                        if c == None:
                            continue
                        if c in code_mapped:
                            code_mapped_query[idx].append(code_mapped[c])
                        else:
                            indices = medical_code_vocabulary.index[medical_code_vocabulary['med_code'] == c].tolist()
                            if len(indices) > 0:
                                code_mapped_query[idx].append(indices[0])
                                code_mapped[c] = indices[0]
                            else:
                                ##remap ICD code
                                for k, v in medical_code_range.items():
                                    if is_in_general_range(d, k):
                                        code_mapped_query[idx].append(v)
                                        code_mapped[c] = v
                                        continue
            else:
                if v is None:
                    continue
                elif len(v) == 0:
                    continue
                else:
                    for c in v:
                        if c == None:
                            continue
                        if c in code_mapped:
                            code_mapped_query[idx].append(code_mapped[c])
                        else:
                            indices = medical_code_vocabulary.index[medical_code_vocabulary['med_code'] == c].tolist()
                            if len(indices) > 0:
                                code_mapped_query[idx].append(indices[0])
                                code_mapped[c] = indices[0]
                            else:
                                ##remap ICD code
                                for k, v in medical_code_range.items():
                                    if is_in_general_range(d, k):
                                        code_mapped_query[idx].append(v)
                                        code_mapped[c] = v
                                        continue
        
    if len(code_mapped_query[idx]) == 0:
        code_mapped_query[idx] = [len(medical_code_vocabulary)]

train_dataset = []
for idx, _ in enumerate(code_mapped_query):
    query = data[idx]
    medical_indices = code_mapped_query[idx]
    print(medical_indices)
    if dataset == 'difficult':
        train_dataset.append({'input': [query['query'], query['answer']], "medical_codes": medical_indices})
    else:
        train_dataset.append({'input': [query['text'], query['answer']], "medical_codes": medical_indices})

with open(dataset + '_dataset.json', 'w') as f:
    json.dump(train_dataset, f)

    
    
    
