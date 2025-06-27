import os
from typing import Optional, List, Dict, Union, Tuple

import pandas as pd

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseEHRDataset
from pyhealth.datasets.utils import strptime

# TODO: add other tables


class EHRShotDataset(BaseEHRDataset):

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "patients.csv"),
            dtype={"patient_id": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "admissions.csv"),
            dtype={"patient_id": str, "visit_id": str},
        )
        # merge patients and admissions tables
        df = pd.merge(patients_df, admissions_df, on="patient_id", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["patient_id", "start", "end"], ascending=True)
        # group by patient
        df_group = df.groupby("patient_id")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            # no exact birth datetime in MIMIC-IV
            # use anchor_year and anchor_age to approximate birth datetime
            patient = Patient(
                patient_id=p_id,
                # no exact month, day, and time, use Jan 1st, 00:00:00
                birth_datetime=strptime(str(p_info['dob'].values[0])),
                # no exact time, use 00:00:00
                death_datetime=strptime(p_info["dod"].values[0]),
                gender=p_info["Gender"].values[0],
                ethnicity=p_info["Race"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("visit_id"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info["start"].values[0]),
                    discharge_time=strptime(v_info["end"].values[0]),
                    visit_type=v_info["visit_type"].values[0]
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.patient_id.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients


    def parse_diagnoses(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        table = "diagnoses"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patient_id": str, "visit_id": str, "code_val": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patient_id", "visit_id", "code_val"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["patient_id", "visit_id", "code_val"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("patient_id")

        # parallel unit of diagnosis (per patient)
        def diagnosis_unit(p_id, p_info):
            events = []
            # iterate over each patient and visit
            for v_id, v_info in p_info.groupby("visit_id"):
                for code, version in zip(v_info["code_val"], v_info["icd_version"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=f"ICD{version}CM",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: diagnosis_unit(x.patient_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients


    def parse_procedures(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        table = "procedures"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patient_id": str, "visit_id": str, "code_val": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patient_id", "visit_id", "code_val", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["patient_id", "visit_id"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("patient_id")

        # parallel unit of procedure (per patient)
        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("visit_id"):
                for code, version in zip(v_info["code_val"], v_info["icd_version"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=f"ICD{version}PROC",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: procedure_unit(x.patient_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients


    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        table = "prescriptions"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"patient_id": str, "visit_id": str, "code_val": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patient_id", "visit_id", "code_val"])
        # sort by start date and end date
        df = df.sort_values(
            ["patient_id", "visit_id", "start", "end"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("patient_id")

        # parallel unit of prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("visit_id"):
                for timestamp, code in zip(v_info["start"], v_info["code_val"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ATC",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: prescription_unit(x.patient_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients


    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        table = "LABEVENTS"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "itemid": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "itemid"])
        # sort by charttime
        df = df.sort_values(["subject_id", "hadm_id", "charttime"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of labevent (per patient)
        def lab_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code in zip(v_info["charttime"], v_info["itemid"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MIMIC4_ITEMID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: lab_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients


def parse_hcpcsevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        table = "HCPCSEVENTS"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "hcpcs_cd": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "hcpcs_cd"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of hcpcsevents (per patient)
        def hcpcsevents_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code in v_info["hcpcs_cd"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MIMIC4_HCPCS_CD",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    # update patients
                    events.append(event)
            return events
            
        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: hcpcsevents_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        
        return patients
