# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools

def get_preprocessed_samsum(dataset_config, tokenizer, split):
    url = "https://objectstorage.us-ashburn-1.oraclecloud.com/p/jEHBWCe-9_cHxJ03M_TVZUd1C9kZn-H2OAZD7VHYt_lv-0WPcgho_VrXc7zvmx3N/n/sehubjapacprod/b/LT_Liang_Datascience/o/dataset/ChatMed_Consult-v0.3.json"
    dataset = datasets.load_dataset("json", data_files=url,split=split)
    dataset = dataset.select([ 0 , 10 , 20 , 30 , 40 , 50 ])
    
    def apply_prompt_template(sample):
        prompt = f"#ASK:\n{{query}}\n\nANSWER:\n{{response}}"
        return {
            "prompt": prompt.format(query=sample["query"],response=sample["response"])
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    
    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)

        sample = {
            "input_ids": prompt
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
