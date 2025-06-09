import torch
import numpy as np
from torch_geometric import data as DATA
from torch_geometric.data import Batch

class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_visits=50, max_medical_code=1000, task = 'mortality', labels=None):
        self.dataset = dataset
        self.max_visits = max_visits
        self.max_medical_code = max_medical_code
        self.task = task
        self.ethnicity_dict = {}
        self.gender_dict = {}
        self.labels = labels

        ##here should be the embedding of medical codes
        self.med_codes_pkg_map_path = '/n/netscratch/mzitnik_lab/Lab/xsu/MultimodalTokenizer/pre_trained_model/2025-01-12-03-47-16-000-GCN_bert-base-uncased/embeddings_all.npy'
        self.pre_trained_embedding = np.load(self.med_codes_pkg_map_path)
    
    def time_format(self, datetime):
        days_in_year = 365  # Approximation; adjust if leap years matter in your context
        years = datetime.days // days_in_year
        remaining_days = datetime.days % days_in_year
        hours = datetime.seconds // 3600

        if years < 0 or remaining_days < 0 or hours < 0:
            return [0, 0, 0]

        return [years, remaining_days, hours]
    
    def calculate_time_interval(self, patient_birthdate, encounter_timestamp, discharge_timestamp):
        
        time_between_visit = np.zeros(shape=(self.max_visits, 3))
        time_within_visit = np.zeros(shape=(self.max_visits, 3))
        
        for i in range(len(discharge_timestamp)):
            time_interval = discharge_timestamp[i] - encounter_timestamp[i]
            time_within_visit[i] = self.time_format(time_interval)

        discharge_timestamp = [patient_birthdate] + discharge_timestamp
    
        for i in range(len(encounter_timestamp)):
            time_interval = encounter_timestamp[i] - discharge_timestamp[i]
            time_between_visit[i] = self.time_format(time_interval)
    
        return time_between_visit, time_within_visit
    
    def get_visit(self, conditions_map, procedures_map, drugs_map):

        assert len(conditions_map) == len(procedures_map) == len(drugs_map)
        ##time_interval_within_visit = np.zeros(shape=(self.max_visits, 3))
        conditions_map = conditions_map[0]
        procedures_map = procedures_map[0]
        drugs_map = drugs_map[0]
        
        codes = []
        visit_order_id = []
        for v_i, _ in enumerate(conditions_map):
            procedures = procedures_map[v_i]
            conditions = conditions_map[v_i]
            if len(drugs_map) > 0:
                if v_i < len(drugs_map):
                    drugs = drugs_map[v_i]
                else:
                    drugs = []
            else:
                drugs = []
            node_set = conditions + procedures + drugs
            node_set = [self.pre_trained_embedding.shape[0] if x == -1 else x for x in node_set]
            visit_order_id.extend([v_i for _ in range(len(node_set))])
            codes.extend(node_set)
    
        ##get_masking_strategy
        code_mask = np.ones(shape=(self.max_medical_code, 1))
        code_mask[:len(codes)] = 0
       
        d = self.max_medical_code - len(codes)
        #print(d)
        pad = [0 for _ in range(d)]
        codes.extend(pad)
        visit_order_id.extend(pad)

        return np.array(codes).reshape(self.max_medical_code, 1), np.array(visit_order_id).reshape(self.max_medical_code, 1), code_mask

    def get_visit_EHRShot(self, codes_map):
        codes = []
        visit_order_id = []
        for v_i, _ in enumerate(codes_map):
            code_visit = codes_map[v_i]
            node_set = code_visit
            node_set = [self.pre_trained_embedding.shape[0] if x == -1 else x for x in node_set]
            visit_order_id.extend([v_i for _ in range(len(node_set))])
            codes.extend(node_set)
    
        ##get_masking_strategy
        code_mask = np.ones(shape=(self.max_medical_code, 1))
        code_mask[:len(codes)] = 0
       
        d = self.max_medical_code - len(codes)
        #print(d)
        pad = [0 for _ in range(d)]
        codes.extend(pad)
        visit_order_id.extend(pad)

        return np.array(codes).reshape(self.max_medical_code, 1), np.array(visit_order_id).reshape(self.max_medical_code, 1), code_mask

    
    def get_data(self, idx):
        data = self.dataset[idx][0]  ##should be a dictionary here

        birthdate = data['birthdate']
        if data['gender'] not in self.gender_dict:
            self.gender_dict[data['gender']] = len(self.gender_dict)
        
        gender_int = self.gender_dict[data['gender']]
        
        
        if data['ethnicity'] not in self.ethnicity_dict:
            self.ethnicity_dict[data['ethnicity']] = len(self.ethnicity_dict)

        ethnicity_int = self.ethnicity_dict[data['ethnicity']]
        
        sorted_encounter_timestamp = data['timestamp_encounter']
        sorted_discharge_timestamp = data['timestamp_discharge'] 
        time_interval_between_visit, time_interval_within_visit = self.calculate_time_interval(birthdate, list(sorted_encounter_timestamp), list(sorted_discharge_timestamp))
        
        #if self.dataset in ['MIMIC_III', 'MIMIC_IV']:
        code_index, visit_id, code_mask = self.get_visit(data['conditions_map'], data['procedures_map'], data['drugs_map'])## should be [max_medical_code]
        #else:
        #    code_index, visit_id, code_mask = self.get_visit_EHRShot(data['codes_map'])## should be [max_medical_code]
        #print(self.labels[idx])
        data = DATA.Data(x=torch.LongTensor([code_index]),
                         visit_id = torch.LongTensor([visit_id]),
                         code_mask = torch.LongTensor([code_mask]),
                         gender = torch.LongTensor([int(gender_int)]),
                         ethnicity = torch.LongTensor([int(ethnicity_int)]),
                         timestamp_within_visits = torch.LongTensor(np.array([time_interval_within_visit])),
                         timestamp_between_visits = torch.LongTensor(np.array([time_interval_between_visit])),
                         label = torch.Tensor([self.labels[idx]]),
                        )
        
        data.__setitem__('c_size', torch.LongTensor([len(code_index)]))
        return data

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.get_data(idx)

def collate(data_list):
    batchA = Batch.from_data_list(data_list)

    return batchA
       