import os
import sys
from typing import List
import fire
import torch
import wandb
import transformers
from transformers import set_seed
from datasets import load_dataset, IterableDataset
from datasets.distributed import split_dataset_by_node
from review import Review
import re
import json
from huggingface_hub import login
torch.cuda.empty_cache()
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(int(os.environ['WORLD_SIZE']))
device = torch.device('cuda', local_rank)
wandb.login()

set_seed(42)

from peft import get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    #prepare_model_for_int8_training,
    set_peft_model_state_dict,
    TaskType
)


def train(
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",  # the only required argument
    data_path: str = "../Dataset/MedicalQA/medmcqa_dataset.json",
    output_dir: str = "./llama_lora_finetune",
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 8,
    num_epochs: int = 20,
    learning_rate: float = 5e-5,
    cutoff_len: int = 256,
    val_set_size: int = 0,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.01,
    lora_target_modules: List[str] = [
        "q_proj",
        #"k_proj",
        "v_proj",
        #"o_proj",
    ],
    num_prefix: int = 1,
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = False,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    #kge_model: str = "pre_train_primekg/prime_rotate_new.pth"
):
    gradient_accumulation_steps = batch_size // micro_batch_size
    print("gradient_accumulation")

    device_map = "auto"
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
   
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    backbone_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    print(backbone_model)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        if not train_on_inputs:
            result["labels"] = [-100] * (len(result["input_ids"]) - 1) + result["labels"][-1:]

        return result

    def build_llama_prompt(system_prompt, user_input, assistant_output=None):
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}

        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if assistant_output is not None:
            text += f"\n{assistant_output}"

        return text

    def generate_and_tokenize_prompt(data_point):
        ##give an example

        query, output = data_point['input'][:2]
        medical_tokens = data_point['medical_codes']

        ins = "The following is a multiple-choice medical question. Please directly select and provide the correct answer from options 'A', 'B, 'C', 'D'. Only return the correct answer by 'A', 'B', 'C', or 'D'."
        question = "The question is: " + query + "\n Answer: \n"

        full_prompt = build_llama_prompt(system_prompt=ins, user_input=question, assistant_output=output)

        medical_tokens_max_length = [0 for _ in range(cutoff_len)]
        medical_tokens_max_length[:len(medical_tokens)] = medical_tokens
        medical_tokens_attention_mask = [0 for _ in range(cutoff_len)]
        medical_tokens_attention_mask[:len(medical_tokens)] = [1 for _ in range(len(medical_tokens))]

        tokenized_full_prompt = tokenize(full_prompt)
        tokenized_full_prompt['input_ids'] = medical_tokens_max_length + tokenized_full_prompt['input_ids']
        tokenized_full_prompt['attention_mask'] = medical_tokens_attention_mask + tokenized_full_prompt['attention_mask']
    
        return tokenized_full_prompt

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(backbone_model, config)
    slama_model = Review(model, num_prefix, hidden_dim=1024)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    
    #print(data)
    num_rows_in_train = len(data)
    print(num_rows_in_train)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint

        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin") 
            resume_from_checkpoint = (False)  # So the trainer won't try loading its state

        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
        val_data = (train_val["test"].shuffle().map(generate_and_tokenize_prompt))
    else:
        train_data = split_dataset_by_node(data["train"], rank=local_rank, world_size=world_size).shuffle(seed=42).map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        print('true!!')
        model.is_parallelizable = True
        model.model_parallel = True
    

    MAX_STEPS = num_epochs *  num_rows_in_train / micro_batch_size
    trainer = transformers.Trainer(
        model=slama_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=100,
            optim="adamw_hf",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=None,
            save_steps=10,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to='wandb',
            run_name='batch_size_{}_gradient_steps_{}_learning_rate_{}_num_epoches_{}_prefix'.format(micro_batch_size, gradient_accumulation_steps, learning_rate, num_epochs),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    torch.save(slama_model.projector, os.path.join(output_dir, "projector.pth"))

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
