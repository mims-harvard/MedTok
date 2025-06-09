<h1 align="center">
  MedTok: Multimodal Medical Code Tokenizer
</h1>

## 👀 Overview of MedTok
Foundation models trained on patient electronic health records (EHRs) require tokenizing medical data into sequences of discrete vocabulary items. Existing tokenizers treat medical codes from EHRs as isolated textual tokens. However, each medical code is defined by its textual description, its position in ontological hierarchies, and its relationships to other codes, such as disease co-occurrences and drug-treatment associations. Medical vocabularies contain more than 600,000 codes with critical information for clinical reasoning. We introduce MedTok, a multimodal medical code tokenizer that uses the text descriptions and relational context of codes. MedTok processes text using a language model encoder and encodes the relational structure with a graph encoder. It then quantizes both modalities into a unified token space, preserving modality-specific and cross-modality information. We integrate MedTok into five EHR models and evaluate it on operational and clinical tasks across in-patient and out-patient datasets, including outcome prediction, diagnosis classification, drug recommendation, and risk stratification. Swapping standard EHR tokenizers with MedTok improves AUPRC across all EHR models, by 4.10% on MIMIC-III, 4.78% on MIMIC-IV, and 11.32% on EHRShot, with the largest gains in drug recommendation. Beyond EHR modeling, we demonstrate using MedTok tokenizer with medical QA systems. Our results demonstrate the potential of MedTok as a unified tokenizer for medical codes, improving tokenization for medical foundation models.



## 🚀 Installation

1⃣️ First, clone the Github repository:

```bash
git clone https://github.com/mims-harvard/MedTok
cd MedTok
```

2⃣️ Then, set up the environment. To create an environment with all of the required packages, please ensure that conda is installed and then execute the commands:

```bash
conda env create -f MedTok.yaml
conda activate MedTok
```

## 💡 How to train MedTok?

After cloning the repository and installing all dependencies. 

## 🛠️ How to use MedTok?


### MedTok with QA Task


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

