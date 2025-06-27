<h1 align="center">
  MedTok: Multimodal Medical Code Tokenizer
</h1>

## 👀 Overview of MedTok
Foundation models trained on patient electronic health records (EHRs) require tokenizing medical data into sequences of discrete vocabulary items. Existing tokenizers treat medical codes from EHRs as isolated textual tokens. However, each medical code is defined by its textual description, its position in ontological hierarchies, and its relationships to other codes, such as disease co-occurrences and drug-treatment associations. Medical vocabularies contain more than 600,000 codes with critical information for clinical reasoning. We introduce MedTok, a multimodal medical code tokenizer that uses the text descriptions and relational context of codes. MedTok processes text using a language model encoder and encodes the relational structure with a graph encoder. It then quantizes both modalities into a unified token space, preserving modality-specific and cross-modality information. We integrate MedTok into five EHR models and evaluate it on operational and clinical tasks across in-patient and out-patient datasets, including outcome prediction, diagnosis classification, drug recommendation, and risk stratification. Swapping standard EHR tokenizers with MedTok improves AUPRC across all EHR models, by 4.10% on MIMIC-III, 4.78% on MIMIC-IV, and 11.32% on EHRShot, with the largest gains in drug recommendation. Beyond EHR modeling, we demonstrate using MedTok tokenizer with medical QA systems. Our results demonstrate the potential of MedTok as a unified tokenizer for medical codes, improving tokenization for medical foundation models.

![MedTok framework](https://github.com/mims-harvard/MedTok/blob/main/MedTok.jpg)

## 🚀 Installation

Clone the Github repository and setup the enviroment.

```bash
git clone https://github.com/mims-harvard/MedTok
cd MedTok
```

```bash
conda env create -f MedTok.yaml
conda activate MedTok
```

## 💡 How to train MedTok?

To train MedTok, please first download [all_codes_mappings.parquet](data_link) to 'Dataset/medicalCode/', [primeKG.bin](data link) to 'Dataset/primeKG/' and then run:

```bash
sbatch run.sh
```

## 🛠️ How to use MedTok?

We provide two ways to use MedTok. One is using this codebase to run inference script to get all tokens, the other is accessing MedTok by [MedTok](add links).

```bash
python inference.py
```
or

```bash
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("MedTok")
tokens = tokenizer.tokenize("E11.9")
ids = tokenizer.encode("E11.9")
embed = tokenizer.embed("E11.9")
```

### 🏥MedTok for EHR
Please first download EHR datasets to 'Dataset/EHR/{EHR_dataset_name}', and then run:
```bash
cd MedTok_EHR_Tutorial
python EHRModel_token.py
```

### ❓MedTok for MedicalQA
To finetune LLMs with datasets we presented in our paper, please run the following command:
```bash
cd MedTok_QA_Tutorial
WORLD_SIZE=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 1234 fintune_llama3.py
```
After obtaining the pre-trained model, please do inference directly on other datasets:
```bash
cd MedTok_QA_Tutorial
WORLD_SIZE=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 1234 inference.py
```

If you want to apply MedTok to your own QA system or datasets, please first extract the diseases contained in each query and obtain their medical code, and then prepare the datasets to be used as training dataset to finetune LLMs.
```bash
cd MedTok_QA_Tutorial
python extract_disease.py
python map_query_id.py
```

## Citation
```bash
@article{su2025multimodal,
  title={Multimodal Medical Code Tokenizer},
  author={Su, Xiaorui and Messica, Shvat and Huang, Yepeng and Johnson, Ruth and Fesser, Lukas and Gao, Shanghua and Sahneh, Faryad and Zitnik, Marinka},
  journal={International Conference on Machine Learning, ICML},
  year={2025}
}
```
</details>

## Contact

If you have any questions or suggestions, please email [Xiaorui Su](xiaorui_su@hms.harvard.edu) and [Marinka Zitnik](marinka@hms.harvard.edu).

