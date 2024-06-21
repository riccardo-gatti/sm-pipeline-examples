# Temporary preprocess step (to be changed with new dataset)
import boto3
from sagemaker.s3_utils import parse_s3_url
import sagemaker
from sagemaker.s3 import S3Uploader
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
from steps.utils import safe_open_w, write_to_file
import json
import os

def preprocess_dft(s3_output_path):
    dataset_path = 'allenai/sciq'
    dataset = load_dataset(dataset_path)

    dataset_training_df = dataset['train'].to_pandas()
    dataset_training_df = dataset_training_df.sample(n=400, random_state=42, ignore_index=True)

    finetuning_dataset_filename = "dataset_finetune_daft.txt"
    evaluation_dataset_filename = "dataset_evaluation.jsonl"
    finetuning_dataset_local_path = f"./output/{dataset_path}/{finetuning_dataset_filename}"
    evaluation_dataset_local_path = f"./output/{dataset_path}/{evaluation_dataset_filename}"

    # Create DAFT dataset
    data_train_daft = " \n".join(((dataset_training_df.drop_duplicates(subset=['support']))['support']))
    write_to_file(data_train_daft, finetuning_dataset_local_path)

    # Create evaluation dataset
    dataset_evaluation_df = dataset_training_df[['question', 'correct_answer']].copy()
    dataset_evaluation_df = dataset_evaluation_df.rename(
        columns={"correct_answer": "target_output", "question": "model_input"})
    dataset_evaluation_df.to_json(evaluation_dataset_local_path, orient="records", lines=True)

    finetuning_dataset_s3_path = f"{s3_output_path}/{dataset_path}/finetuning/dft"
    evaluation_dataset_s3_path = f"{s3_output_path}/{dataset_path}/evaluation"
    print("Uploading finetuning dataset...")
    S3Uploader.upload(finetuning_dataset_local_path, f"{finetuning_dataset_s3_path}")
    print("Uploading evaluation dataset...")
    S3Uploader.upload(evaluation_dataset_local_path, f"{evaluation_dataset_s3_path}")

    return {"s3_output_path": s3_output_path,
            "s3_finetune_dataset_path": finetuning_dataset_s3_path,
            "s3_evaluation_data_location": f"{evaluation_dataset_s3_path}/{evaluation_dataset_filename}"}

def preprocess_ist(s3_output_path):
    dataset_path = 'allenai/sciq'
    dataset = load_dataset(dataset_path)

    dataset_training_df = dataset['train'].to_pandas()
    dataset_training_df = dataset_training_df.sample(n=400, random_state=42, ignore_index=True)

    finetuning_dataset_filename = "dataset_finetune_ist.jsonl"
    template_filename = "template.json"
    evaluation_dataset_filename = "dataset_evaluation.jsonl"
    finetuning_dataset_local_path = f"./output/{dataset_path}/{finetuning_dataset_filename}"
    evaluation_dataset_local_path = f"./output/{dataset_path}/{evaluation_dataset_filename}"
    template_local_path = f"./output/{dataset_path}/{template_filename}"

    # Write template file
    template = {
        "prompt": "You are an expert answering science related question.\n\n### Answer this question:\n{question}\n",
        "completion": "{correct_answer}"
    }
    with safe_open_w(template_local_path) as text_file:
        json.dump(template, text_file)

    # Create IST dataset
    dataset_train_ist_df = dataset_training_df[['question', 'correct_answer']]
    dataset_train_ist_df.to_json(finetuning_dataset_local_path, orient="records", lines=True)

    # Create evaluation dataset
    dataset_evaluation_df = dataset_training_df[['question', 'correct_answer']].copy()
    dataset_evaluation_df[
        "question"] = "You are an expert answering science related question.\n\n### Answer this question:\n" + \
                      dataset_evaluation_df["question"].astype(str)
    dataset_evaluation_df = dataset_evaluation_df.rename(
        columns={"correct_answer": "target_output", "question": "model_input"})
    dataset_evaluation_df.to_json(evaluation_dataset_local_path, orient="records", lines=True)

    finetuning_dataset_s3_path = f"{s3_output_path}/{dataset_path}/finetuning/ist"
    evaluation_dataset_s3_path = f"{s3_output_path}/{dataset_path}/evaluation"

    print("Uploading finetuning dataset...")
    S3Uploader.upload(finetuning_dataset_local_path, f"{finetuning_dataset_s3_path}")
    print("Uploading template...")
    S3Uploader.upload(template_local_path, f"{finetuning_dataset_s3_path}")
    print("Uploading evaluation dataset...")
    S3Uploader.upload(evaluation_dataset_local_path, f"{evaluation_dataset_s3_path}")

    return {"s3_output_path": s3_output_path,
            "s3_finetune_dataset_path": finetuning_dataset_s3_path,
            "s3_evaluation_data_location": f"{evaluation_dataset_s3_path}/{evaluation_dataset_filename}"}

