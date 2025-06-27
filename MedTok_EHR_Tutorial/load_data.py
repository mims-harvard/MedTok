import pickle
from tqdm import tqdm
import numpy as np
import torch
from dgl.data.utils import save_graphs
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.data import Data
from tqdm import tqdm
import multiprocessing
from data import Event, Visit, Patient
from mimic3 import MIMIC3Dataset
from mimic4 import MIMIC4Dataset
from ehrshot import EHRShotDataset
from pyhealth.datasets import BaseEHRDataset
from pyhealth.datasets.utils import strptime
from pyhealth.tasks import drug_recommendation_mimic3_fn
from pyhealth.medcode import InnerMap
from pyhealth.datasets.utils import *
import pandas as pd
import os
import spacy
from scispacy.linking import EntityLinker
import pandas as pd
import numpy as np
import difflib
import nmslib
from torch_geometric.utils import k_hop_subgraph, to_undirected
from multiprocessing.pool import ThreadPool as Pool
from torch_geometric.data import Data
import networkx as nx
from tqdm import tqdm
import pickle


import torch.nn.functional as F
#import neptune
from copy import deepcopy
import pandas as pd
import pickle
import os
from torch.nn.utils.rnn import pad_sequence
import wandb

root = "../Dataset/EHR"  ##the root directory to save the processed EHR data and access the EHR data
med_codes_pkg_map_path = '../Dataset/medicalCode/all_codes_mappings_v3.parquet'

##read patient EHR data for MIMIC III or MIMIC IV and then obtain the corresponding format for each of tasks, and then generate patient-specific graph for each patient for each visit
class PatientEHR(object):
    def __init__(self, dataset, split, visit_num_th, max_visit_th, task='mortality', remove_outliers=True):
        super(PatientEHR, self).__init__()

        self.dataset = dataset
        self.split = split
        self.visit_num_th = visit_num_th
        self.max_visit_th = max_visit_th
        self.task = task
        self.is_remove = remove_outliers

        self.medical_code = pd.read_parquet(med_codes_pkg_map_path)
        self.medical_code['med_code'] = self.medical_code['med_code'].apply(lambda x: x.replace('.', ''))
        #print(self.medical_code.head())
        self.medical_code_range = {}
        for idx, m in enumerate(self.medical_code['med_code']):
            if '-' in m and '.' in m:
                self.medical_code_range[m] = idx
        
        #print(self.medical_code.head())
        
        self.root = root
        self.condition_dict = {}
        self.procedure_dict = {}
        self.drug_dict = {}

        if os.path.exists(os.path.join(self.root, f"{dataset}",f"{dataset}_patient_{task}.pkl")):
            with open(os.path.join(self.root, f"{dataset}", f"{dataset}_patient_{task}.pkl"), 'rb') as f:
                self.patient_ehr_data = pickle.load(f)
        else:
            self.database = self.load_database()
            self.patient_ehr_data = self.process_structure_EHR_for_patient()
    
    def load_database(self):
        ##the structure EHR data are consctructed by patient, dianosis, procedures, prescriptions, and labevents tables
        ##using Pyhealth package get data in these two EHR datasets
        if self.dataset == 'MIMIC_III':
            database = MIMIC3Dataset(
                root = os.path.join(root, "MIMIC_III"),
                tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
                dev=False,
                code_mapping={
                    "NDC": ("ATC", {"target_kwargs": {"level": 5}})
                    },   
                #refresh_cache=True     
            )
        elif self.dataset == 'MIMIC_IV':
            database = MIMIC4Dataset(
                root = os.path.join(root, "MIMIC_IV"),
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                dev=False,
                code_mapping={
                    "NDC": ("ATC", {"target_kwargs": {"level": 5}})
                    },
                refresh_cache=True
            )
        elif self.dataset == 'EHRShot':
            print("dataset", self.dataset)
            database = EHRShotDataset(
                root = os.path.join(root, "EHRShot"),
                tables=["diagnoses", "procedures", "prescriptions"],
                dev=False,
                code_mapping={
                    "NDC": ("ATC", {"target_kwargs": {"level": 5}})
                    },
                refresh_cache=True
            )
    
        return database
    
    def process_structure_EHR_for_patient(self):
        ##process the structure EHR data for each patient
        samples = []

        disease_candidate = ['4100', '4101', '4102', '4103', '4104', '4105', '4106', '4107', '4108', '4109', 'I210', 'I211', 'I213', 'I214', 'I219', 'I22x', '41000', '41001', '41010', '41011', '41020', '41021', '41030', '41031', '41040', '41041', '41050', '41051', '41060', '41061', '41070', '41071', '41080', '41081', '41090', '41091'] ##AMI
        disease_candidate_index = []
        for d in disease_candidate:
            mapped_indicies = self.medical_code.index[self.medical_code['med_code'] == d].tolist()
            if len(mapped_indicies) > 0:
                disease_candidate_index.extend(mapped_indicies)
            else:
                ##remap ICD code
                for k, v in self.medical_code_range.items():
                    if self.is_in_general_range(d, k):
                        disease_candidate_index.append(ValueError)
                        continue
                    

        print("processing EHR data")
        for _, patient in tqdm(self.database.patients.items(), desc = "Processing EHR data"):
            if self.task == 'mortality':
                sample = self.mortality_dataset(patient)
                if sample is not None:
                    samples.append(sample)
            elif self.task == 'readmission':
                sample = self.readmission_dataset(patient)
                if sample is not None:
                    samples.append(sample)
            elif self.task == 'lenofstay':
                sample = self.length_of_stay_dataset(patient)
                if sample is not None:
                    samples.append(sample)
            elif self.task == 'phenotype':
                self.phenotype_index = pd.read_pickle('phenotype_index.pkl')
                sample = self.phenotype_dataset(patient)
                if sample is not None:
                    samples.append(sample)
            elif self.task == 'drugrec':
                sample = self.drugrec_dataset(patient)
                if sample is not None:
                    samples.append(sample)
            elif self.task == 'new_disease':
                sample = self.new_assignment_diseases(patient, disease_candidate_index)
                if sample is not None:
                    samples.append(sample)
        
        file_path = os.path.join(self.root, f"{self.dataset}_patient_{self.task}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(samples, f)

        return samples
    
    def readmission_dataset_ehrshot(self, p_info, p_visit): ##30 days readmission
        samples = []
        visit_time = p_visit['visit_timestamp']
        visit_type = p_visit['visit_type']
        death = p_visit['death']
        codes = p_visit['codes']

        codes_selected = []
        time_selected = []

        for time, v_t, d, c in zip(visit_time, visit_type, death, codes):
            if len(c) > 0:
                time_selected.append(time)
                codes_selected.append(c)
        
        if len(time_selected) > 2:
            v_t_t1 = visit_time[-2]
            v_t_t2 = visit_time[-1]
            if (v_t_t2 - v_t_t1).days <= 30:
                label = 1
            else:
                label = 0
        else:
            return None
        
        if 'Ethnicity' in p_info:
            ethnicity = p_info['Ethnicity']
        else:
            ethnicity = 'Unknown'
        samples.append(
                {
                    "patient_id": p_info['p_tid'],
                    "birthdate": p_info['birth_year'],
                    "deathdate": None,
                    "gender": p_info['Gender'],
                    "ethnicity": ethnicity,
                    "codes_map": codes_selected[:-1],
                    "label": label,
                    "timestamp_encounter": time_selected[:-1],
                    "timestamp_discharge": time_selected[:-1],
                }
            )
        
        if len(time_selected) > 500:
            return None
        
        print(samples)

        return samples 
    
    def mortality_dataset_ehrshot(self, p_info, p_visit):
        #print(p_info)
        samples = []
        visit_time = p_visit['visit_timestamp']
        visit_type = p_visit['visit_type']
        death = p_visit['death']
        codes = p_visit['codes']

        codes_selected = []
        time_selected = []
        for time, v_t, d, c in zip(visit_time, visit_type, death, codes):
            if len(c) > 0:
                time_selected.append(time)
                codes_selected.append(c)
        if 'Ethnicity' in p_info:
            ethnicity = p_info['Ethnicity']
        else:
            ethnicity = 'Unknown'
        samples.append(
                {
                    "patient_id": p_info['p_tid'],
                    "birthdate": p_info['birth_year'],
                    "deathdate": None,
                    "gender": p_info['Gender'],
                    "ethnicity": ethnicity,
                    "codes_map": codes_selected[:-1],
                    "label": death[-1],
                    "timestamp_encounter": time_selected[:-1],
                    "timestamp_discharge": time_selected[:-1],
                }
            )
        
        print(samples)

        return samples 

    def is_in_general_range(self, value, range_string):
        import re
        #print(value, range_string)
        left, right = range_string.split('-')[:2]
        #print(left, right)
        if value >= left and value <= right:
            return True
        else:
            return False

    def sorted_visit(self, patient: Patient):
        conditions_map_all = []
        procedures_map_all = []
        drugs_map_all = []
        visit_encounter_time = []
        visit_discharge_time = []
        
        for i in range(len(patient)):
            visit: Visit = patient[i]
            #print(visit)
            if self.dataset == 'EHRShot':
                conditions = visit.get_code_list(table="diagnoses")
                procedures = visit.get_code_list(table="procedures")
                drugs = visit.get_code_list(table="prescriptions")
            else:
                conditions = [c.replace('.', '') for c in visit.get_code_list(table="DIAGNOSES_ICD")]
                procedures = visit.get_code_list(table="PROCEDURES_ICD")
                drugs = visit.get_code_list(table="PRESCRIPTIONS")

            conditions_map = []
            procedures_map = []
            drugs_map = []

            # exclude: visits without condition, procedure, and drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            for c in conditions:
                if c in self.condition_dict:
                    conditions_map.append(self.condition_dict[c])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == c].tolist()
                if len(indices) > 0:
                    conditions_map.append(indices[0])
                    self.condition_dict[c] = indices[0]
                else:
                    ##remap ICD code
                    for k, v in self.medical_code_range.items():
                        if self.is_in_general_range(c, k):
                            conditions_map_ = v
                            conditions_map.append(conditions_map_)
                            self.condition_dict[c] = conditions_map_
                            continue 
                        else:
                            conditions_map.append(-1)

            for p in procedures:
                if p in self.procedure_dict:
                    procedures_map.append(self.procedure_dict[p])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == p].tolist()
                if len(indices) > 0:
                    procedures_map.append(indices[0])
                    self.procedure_dict[p] = indices[0]
                else:
                    for k, v in self.medical_code_range.items():
                        if self.is_in_general_range(p, k):
                            procedures_map.append(v)
                            self.procedure_dict[p] = v
                            continue
                        else:
                            procedures_map.append(-1)
            
            for d in drugs:
                if d in self.drug_dict:
                    drugs_map.append(self.drug_dict[d])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == d].tolist()
                if len(indices) > 0:
                    drugs_map.append(indices[0])
                    self.drug_dict[d] = indices[0]
                else:
                    drugs_map.append(-1)

            visit_encounter_time.append(visit.encounter_time)
            visit_discharge_time.append(visit.discharge_time)
            conditions_map_all.append(list(set(conditions_map)))
            procedures_map_all.append(list(set(procedures_map)))
            drugs_map_all.append(list((drugs_map)))

        return visit_encounter_time, visit_discharge_time, conditions_map_all, procedures_map_all, drugs_map_all
    
    def readmission_dataset_current(self, patient: Patient, time_window=15):

        if len(patient) < self.visit_num_th:
            return None
    
        samples = []
        for i in range(len(patient)-1): ##here the len means the number of visits
            visit: Visit = patient[i]
            next_visit: Visit = patient[i+1]

            time_diff = (next_visit.encounter_time - visit.encounter_time).days
            readmission_label = 1 if time_diff <= time_window else 0
            
            conditions = visit.get_code_list(table="diagnoses_icd")
            procedures = visit.get_code_list(table="procedures_icd")
            drugs = visit.get_code_list(table="prescriptions")

            # get code index in our medical code vocabulary
            conditions_map = []
            procedures_map = []
            drugs_map = []
            for c in conditions:
                if c in self.condition_dict:
                    conditions_map.append(self.condition_dict[c])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == c].tolist()
                if len(indices) > 0:
                    conditions_map.append(indices[0])
                    self.condition_dict[c] = indices[0]
                else:
                    ##remap ICD code
                    for k, v in self.medical_code_range.items():
                        if self.is_in_general_range(c, k):
                            conditions_map_ = v
                            conditions_map.append(conditions_map_)
                            self.condition_dict[c] = conditions_map_
                            continue
                        else:
                            conditions_map.append(-1)

            for p in procedures:
                if p in self.procedure_dict:
                    procedures_map.append(self.procedure_dict[p])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == p].tolist()
                if len(indices) > 0:
                    procedures_map.append(indices[0])
                    self.procedure_dict[p] = indices[0]
                else:
                    for k, v in self.medical_code_range.items():
                        if self.is_in_general_range(p, k):
                            procedures_map.append(v)
                            self.procedure_dict[p] = v
                            continue
                        else:
                            procedures_map.append(-1)
            
            for d in drugs:
                if d in self.drug_dict:
                    drugs_map.append(self.drug_dict[d])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == d].tolist()
                if len(indices) > 0:
                    drugs_map.append(indices[0])
                    self.drug_dict[d] = indices[0]
                else:
                    drugs_map.append(-1)

            # exclude: visits without condition, procedure, or drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue
            
            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "birthdate": patient.birth_datatime,
                    "deathdate": patient.death_datetime,
                    "gender": patient.gender,
                    "ethnicity": patient.ethnicity,
                    "conditions": [conditions],
                    "procedures": [procedures],
                    "drugs": [drugs],
                    "conditions_map": [conditions_map],
                    "procedures_map": [procedures_map],
                    "drugs_map": [drugs_map],
                    "label": readmission_label,
                    "timestamp_encounter": visit.encounter_time,
                    "timestamp_discharge": visit.discharge_time,
                }
        )
    
        return samples

    def readmission_dataset(self, patient: Patient, time_window=15):
        samples = []
        visit_encounter_time, visit_discharge_time, conditions_map_all, procedures_map_all, drugs_map_all = self.sorted_visit(patient)
        
        if len(visit_encounter_time) < 2:
            return None
        
        data_pairs = list(zip(visit_encounter_time, visit_discharge_time, conditions_map_all, procedures_map_all, drugs_map_all))
        sorted_data_pairs = sorted(data_pairs, key=lambda x: x[0])
        sorted_encounter_time, sorted_discharge_time, sorted_conditions_map, sorted_procedures_map, sorted_drugs_map = zip(*sorted_data_pairs)

        for i in range(len(sorted_encounter_time)-1):
            time_diff = (sorted_encounter_time[i+1] - sorted_encounter_time[i]).days
            readmission_label = 1 if time_diff <= time_window else 0


            # TODO: should also exclude visit with age < 18
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "birthdate": patient.birth_datetime,
                    "deathdate": patient.death_datetime,
                    "gender": patient.gender,
                    "ethnicity": patient.ethnicity,
                    "conditions_map": [sorted_conditions_map[:i+1]],
                    "procedures_map": [sorted_procedures_map[:i+1]],
                    "drugs_map": [sorted_drugs_map[:i+1]],
                    "label": readmission_label,
                    "timestamp_encounter": sorted_encounter_time[:i+1],
                    "timestamp_discharge": sorted_discharge_time[:i+1],
                }
            )
        # no cohort selection
        return samples

    def mortality_dataset(self, patient: Patient):
        samples = []
        conditions_map_all = []
        procedures_map_all = []
        drugs_map_all = []
        visit_encounter_time = []
        visit_discharge_time = []
        mortality_label_status = []
    
        if len(patient) < self.visit_num_th:
            return None
        
        for i in range(len(patient)):
            visit: Visit = patient[i]
            if self.dataset in ['MIMIC_III', 'MIMIC_IV']:
                conditions = visit.get_code_list(table="DIAGNOSES_ICD")
                procedures = visit.get_code_list(table="PROCEDURES_ICD")
                drugs = visit.get_code_list(table="PRESCRIPTIONS")
            else:
                conditions = visit.get_code_list(table="diagnoses")
                procedures = visit.get_code_list(table="procedures")
                drugs = visit.get_code_list(table="prescriptions")
                
            conditions_map = []
            procedures_map = []
            drugs_map = []

            # exclude: visits without condition, procedure, and drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            for c in conditions:
                if c in self.condition_dict:
                    conditions_map.append(self.condition_dict[c])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == c].tolist()
                if len(indices) > 0:
                    conditions_map.append(indices[0])
                    self.condition_dict[c] = indices[0]
                else:
                    ##remap ICD code
                    for k, v in self.medical_code_range.items():
                        if self.is_in_general_range(c, k):
                            conditions_map.append(v)
                            self.condition_dict[c] = v
                            continue
                        else:
                            conditions_map.append(-1)
                            

            for p in procedures:
                if p in self.procedure_dict:
                    procedures_map.append(self.procedure_dict[p])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'].replace(',', '') == p].tolist()
                if len(indices) > 0:
                    procedures_map.append(indices[0])
                    self.procedure_dict[p] = indices[0]
                else:
                    for k, v in self.medical_code_range.items():
                        if self.is_in_general_range(p, k):
                            procedures_map.append(v)
                            self.procedure_dict[p] = v
                            continue
                        else:
                            procedures_map.append(-1)
            
            for d in drugs:
                if d in self.drug_dict:
                    drugs_map.append(self.drug_dict[d])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == d].tolist()
                if len(indices) > 0:
                    drugs_map.append(indices[0])
                    self.drug_dict[d] = indices[0]
                else:
                    drugs_map.append(-1)

            visit_encounter_time.append(visit.encounter_time)
            visit_discharge_time.append(visit.discharge_time)
            conditions_map_all.append(list(set(conditions_map)))
            procedures_map_all.append(list(set(procedures_map)))
            drugs_map_all.append(list(set(drugs_map)))
            mortality_label_status.append(visit.discharge_status)

        if len(visit_encounter_time) < 2:
            return None
        data_pairs = list(zip(visit_encounter_time, visit_discharge_time, conditions_map_all, procedures_map_all, drugs_map_all, mortality_label_status))
        sorted_data_pairs = sorted(data_pairs, key=lambda x: x[0])
        sorted_encounter_time, sorted_discharge_time, sorted_conditions_map, sorted_procedures_map, sorted_drugs_map, sorted_mortality_label_status = zip(*sorted_data_pairs)
        
        if self.dataset in ['MIMIC_III', 'MIMIC_IV']:
            for i in range(len(sorted_encounter_time)-1):
                if sorted_mortality_label_status[i+1] not in [0, 1]:
                    mortality_label = 0
                else:
                    mortality_label = int(sorted_mortality_label_status[i+1])


                # TODO: should also exclude visit with age < 18
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "birthdate": patient.birth_datetime,
                        "deathdate": patient.death_datetime,
                        "gender": patient.gender,
                        "ethnicity": patient.ethnicity,
                        "conditions_map": [sorted_conditions_map[:i+1]],
                        "procedures_map": [sorted_procedures_map[:i+1]],
                        "drugs_map": [sorted_drugs_map[:i+1]],
                        "label": mortality_label,
                        "timestamp_encounter": sorted_encounter_time[:i+1],
                        "timestamp_discharge": sorted_discharge_time[:i+1],
                    }
                )
        elif self.dataset == 'EHRShot':
            if patient.death_datetime is not None:
                mortality_label = 1
            else:
                mortality_label = 0

            samples.append(
                    {
                    "patient_id": patient.patient_id,
                    "birthdate": patient.birth_datetime,
                    "deathdate": patient.death_datetime,
                    "gender": patient.gender,
                    "ethnicity": patient.ethnicity,
                    "conditions_map": [sorted_conditions_map],
                    "procedures_map": [sorted_procedures_map],
                    "drugs_map": [sorted_drugs_map],
                    "label": mortality_label,
                    "timestamp_encounter": sorted_encounter_time,
                    "timestamp_discharge": sorted_discharge_time,
                }
            )
        # no cohort selection
        return samples
    
    def new_assignment_diseases(self, patient: Patient, disease_candidate_index):
        samples = []
        visit_encounter_time, visit_discharge_time, conditions_map_all, procedures_map_all, drugs_map_all = self.sorted_visit(patient)
        #print("new_assignment_diseases")
        if len(visit_encounter_time) < 2:
            return None
        data_pairs = list(zip(visit_encounter_time, visit_discharge_time, conditions_map_all, procedures_map_all, drugs_map_all))
        sorted_data_pairs = sorted(data_pairs, key=lambda x: x[0])
        sorted_encounter_time, sorted_discharge_time, sorted_conditions_map, sorted_procedures_map, sorted_drugs_map = zip(*sorted_data_pairs)

        for i in range(len(sorted_encounter_time)-1):  ##given previous conditions and procedures and drug history, to predict if the drug candidiate will be used in this time
            disease_i = sorted_conditions_map[i+1] + sorted_procedures_map[i+1] + sorted_drugs_map[i+1]
            disease_old = sorted_conditions_map[:i-1]
            label = 0
            time_i = sorted_encounter_time[i]
            time_i_1 = sorted_encounter_time[i+1]
            #print(disease_i)
            for idx, d in enumerate(disease_candidate_index):
                if d in disease_i and (time_i_1 - time_i).days <= 365:
                    label = 1
                    print("positive")
                    continue
                
            # TODO: should also exclude visit with age < 18
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "birthdate": patient.birth_datetime,
                    "deathdate": patient.death_datetime,
                    "gender": patient.gender,
                    "ethnicity": patient.ethnicity,
                    "conditions_map": [sorted_conditions_map[:i+1]],
                    "procedures_map": [sorted_procedures_map[:i+1]],
                    "drugs_map": [sorted_drugs_map[:i+1]],
                    "label": label,
                    "timestamp_encounter": sorted_encounter_time[:i+1],
                    "timestamp_discharge": sorted_discharge_time[:i+1],
                }
            )
        # no cohort selection
        return samples
        
    
    def length_of_stay_dataset(self, patient: Patient):
        #print("length of stay")
        #print(patient)
        samples = []
        visit_encounter_time, visit_discharge_time, conditions_map_all, procedures_map_all, drugs_map_all = self.sorted_visit(patient)
        
        if len(visit_encounter_time) < 1:
            return None
        data_pairs = list(zip(visit_encounter_time, visit_discharge_time, conditions_map_all, procedures_map_all, drugs_map_all))
        sorted_data_pairs = sorted(data_pairs, key=lambda x: x[0])
        sorted_encounter_time, sorted_discharge_time, sorted_conditions_map, sorted_procedures_map, sorted_drugs_map = zip(*sorted_data_pairs)
        #print("length of stay")
        def categorize_los(days):
            if self.dataset == 'EHRShot':
                if days > 0 and days <= 7:
                    return 0
                elif days >7:
                    return 1
            else:
                if days < 1:
                    return 0
                elif 1 <= days <= 7:
                    return days
                elif 8 <= days <= 14:
                    return 8
                else:
                    return 9

        for i in range(len(sorted_encounter_time)-1):
            los_days = (sorted_discharge_time[i] - sorted_encounter_time[i]).days
            #print(los_days)
            los_category = categorize_los(los_days)

            # TODO: should also exclude visit with age < 18
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "birthdate": patient.birth_datetime,
                    "deathdate": patient.death_datetime,
                    "gender": patient.gender,
                    "ethnicity": patient.ethnicity,
                    "conditions_map": [sorted_conditions_map[:i+1]],
                    "procedures_map": [sorted_procedures_map[:i+1]],
                    "drugs_map": [sorted_drugs_map[:i+1]],
                    "label": los_category,
                    "timestamp_encounter": sorted_encounter_time[:i+1],
                    "timestamp_discharge": sorted_discharge_time[:i+1],
                }
            )
        # no cohort selection
        return samples
    
    def phenotype_dataset(self, patient: Patient):
        samples = []
        conditions_map_all = []
        procedures_map_all = []
        drugs_map_all = []
        visit_encounter_time = []
        visit_discharge_time = []
        phenotype_labels = []
    
        if len(patient) < self.visit_num_th:
            return None
        
        for i in range(len(patient)):
            visit: Visit = patient[i]
            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs = visit.get_code_list(table="PRESCRIPTIONS")
            try:
                icu_stay = visit.attr_dict['icustays_num']
            except KeyError:
                print('error')

            if icu_stay > 1:
                continue

            conditions_map = []
            procedures_map = []
            drugs_map = []

            # exclude: visits without condition, procedure, and drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            for c in conditions:
                if c in self.condition_dict:
                    conditions_map.append(self.condition_dict[c])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == c].tolist()
                if len(indices) > 0:
                    conditions_map.append(indices[0])
                    self.condition_dict[c] = indices[0]
                else:
                    ##remap ICD code
                    for k, v in self.medical_code_range.items():
                        if self.is_in_general_range(c, k):
                            conditions_map_ = v
                            conditions_map.append(conditions_map_)
                            self.condition_dict[c] = conditions_map_
                            continue
                        else:
                            conditions_map.append(-1)

            for p in procedures:
                if p in self.procedure_dict:
                    procedures_map.append(self.procedure_dict[p])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == p].tolist()
                if len(indices) > 0:
                    procedures_map.append(indices[0])
                    self.procedure_dict[p] = indices[0]
                else:
                    for k, v in self.medical_code_range.items():
                        if self.is_in_general_range(p, k):
                            procedures_map.append(v)
                            self.procedure_dict[p] = v
                            continue
                        else:
                            procedures_map.append(-1)
            
            for d in drugs:
                if d in self.drug_dict:
                    drugs_map.append(self.drug_dict[d])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == d].tolist()
                if len(indices) > 0:
                    drugs_map.append(indices[0])
                    self.drug_dict[d] = indices[0]
                else:
                    drugs_map.append(-1)
            
            #p_labels = [0] * 25 ## 25 types of phenotypes
            p_labels = []
            node_index = []
            for c_indices in conditions_map:
                node_index.extend(list(self.medical_code['pkg_index_list'][c_indices].values.tolist()))
                
            node_index = np.concatenate(node_index).tolist()
            for l in self.phenotype_index:
                if self.phenotype_index[l] in node_index:
                    p_labels.append(l)
                    #p_labels[self.phenotype_index[l]] = 1
            if len(p_labels) == 0:
                return None
            
            visit_encounter_time.append(visit.encounter_time)
            visit_discharge_time.append(visit.discharge_time)
            conditions_map_all.append(list(set(conditions_map)))
            procedures_map_all.append(list(set(procedures_map)))
            drugs_map_all.append(list(set(drugs_map)))
            phenotype_labels.append(p_labels)
            #mortality_label_status.append(visit.discharge_status)

        if len(visit_encounter_time) < 2:
            return None
        data_pairs = list(zip(visit_encounter_time, visit_discharge_time, conditions_map_all, procedures_map_all, drugs_map_all, phenotype_labels))
        sorted_data_pairs = sorted(data_pairs, key=lambda x: x[0])
        sorted_encounter_time, sorted_discharge_time, sorted_conditions_map, sorted_procedures_map, sorted_drugs_map, sorted_phenotype_labels = zip(*sorted_data_pairs)
        
        for i in range(len(sorted_encounter_time)-1):

            # TODO: should also exclude visit with age < 18
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "birthdate": patient.birth_datetime,
                    "deathdate": patient.death_datetime,
                    "gender": patient.gender,
                    "ethnicity": patient.ethnicity,
                    "conditions_map": [sorted_conditions_map[:i+1]],
                    "procedures_map": [sorted_procedures_map[:i+1]],
                    "drugs_map": [sorted_drugs_map[:i+1]],
                    "label": phenotype_labels[i],
                    "timestamp_encounter": sorted_encounter_time[:i+1],
                    "timestamp_discharge": sorted_discharge_time[:i+1],
                }
            )
        # no cohort selection
        return samples
    
    def drugrec_dataset(self, patient: Patient):
        samples = []
        conditions_map_all = []
        procedures_map_all = []
        drugs_map_all = []
        visit_encounter_time = []
        visit_discharge_time = []

        if len(patient) < self.visit_num_th:
            return None
        
        for i in range(len(patient)):
            visit: Visit = patient[i]
            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs = visit.get_code_list(table="PRESCRIPTIONS")
            try:
                icu_stay = visit.attr_dict['icustays_num']
            except KeyError:
                print('error')

            if icu_stay > 1:
                continue

            conditions_map = []
            procedures_map = []
            drugs_map = []

            # exclude: visits without condition, procedure, and drug code
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            for c in conditions:
                if c in self.condition_dict:
                    conditions_map.append(self.condition_dict[c])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == c].tolist()
                if len(indices) > 0:
                    conditions_map.append(indices[0])
                    self.condition_dict[c] = indices[0]
                else:
                    ##remap ICD code
                    for k, v in self.medical_code_range.items():
                        if self.is_in_general_range(c, k):
                            conditions_map_ = v
                            conditions_map.append(conditions_map_)
                            self.condition_dict[c] = conditions_map_
                            continue
                        else:
                            conditions_map.append(-1)

            for p in procedures:
                if p in self.procedure_dict:
                    procedures_map.append(self.procedure_dict[p])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == p].tolist()
                if len(indices) > 0:
                    procedures_map.append(indices[0])
                    self.procedure_dict[p] = indices[0]
                else:
                    for k, v in self.medical_code_range.items():
                        if self.is_in_general_range(p, k):
                            procedures_map.append(v)
                            self.procedure_dict[p] = v
                            continue
                        else:
                            procedures_map.append(-1)
            
            for d in drugs:
                if d in self.drug_dict:
                    drugs_map.append(self.drug_dict[d])
                    continue
                indices = self.medical_code.index[self.medical_code['med_code'] == d].tolist()
                if len(indices) > 0:
                    drugs_map.append(indices[0])
                    self.drug_dict[d] = indices[0]
                else:
                    drugs_map.append(-1)
            
            visit_encounter_time.append(visit.encounter_time)
            visit_discharge_time.append(visit.discharge_time)
            conditions_map_all.append(list(set(conditions_map)))
            procedures_map_all.append(list(set(procedures_map)))
            drugs_map_all.append(list(set(drugs_map)))

        if len(visit_encounter_time) < 2:
            return None
        data_pairs = list(zip(visit_encounter_time, visit_discharge_time, conditions_map_all, procedures_map_all, drugs_map_all))
        sorted_data_pairs = sorted(data_pairs, key=lambda x: x[0])
        sorted_encounter_time, sorted_discharge_time, sorted_conditions_map, sorted_procedures_map, sorted_drugs_map = zip(*sorted_data_pairs)

        drug_candidate = ['J01XA01', 'J01MA12', 'B01AB01', 'C07AB02', 'C10AA05']
        drug_candidate_index = [self.medical_code.index[self.medical_code['med_code'] == d].tolist()[0] for d in drug_candidate]
        print("drug_candidate_index", drug_candidate_index)
        print(drug_candidate_index)
        
        for i in range(len(sorted_encounter_time)):  ##given previous conditions and procedures and drug history, to predict if the drug candidiate will be used in this time

            drugs_i = sorted_drugs_map[i]
            label = [] 
            for idx, d in enumerate(drug_candidate_index):
                print(d)
                if d in drugs_i:
                    label.append(idx)
            if len(label) == 0:
                continue   
            # TODO: should also exclude visit with age < 18
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "birthdate": patient.birth_datetime,
                    "deathdate": patient.death_datetime,
                    "gender": patient.gender,
                    "ethnicity": patient.ethnicity,
                    "conditions_map": [sorted_conditions_map[:i+1]],
                    "procedures_map": [sorted_procedures_map[:i+1]],
                    "drugs_map": [sorted_drugs_map[:i]],
                    "label": label,
                    "timestamp_encounter": sorted_encounter_time[:i+1],
                    "timestamp_discharge": sorted_discharge_time[:i+1],
                }
            )
        # no cohort selection
        return samples