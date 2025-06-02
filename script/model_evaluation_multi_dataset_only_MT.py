import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
import json
from tqdm import tqdm

from sacrebleu.metrics import BLEU
from tqdm import tqdm


file_path = "/home/llama/Personal_Directories/srb/temperature_eval/data/output/vllm_exp_dataset_csv_Llama-3.2-3B-Instruct-awq__20241226_023606.jsonl"
df = pd.read_json(file_path, lines=True)


model_name = "Llama-3.2-3B-Instruct"


evaluator_model_path = "/home/llama/models/base_models/Llama-3.3-70B-Instruct"
evaluator_model_name = "Llama-3.3-70B-Instruct"

model_name = os.path.splitext(os.path.basename(file_path))[0]


output_folder = (
    "/home/llama/Personal_Directories/srb/temperature_eval/data/output/evaluation/"
)
output_prefix = f"evaluated_{evaluator_model_name}_{model_name}"

bleu = BLEU(tokenize="flores101", effective_order=True)


def compute_bleu_score(row):
    generated_response = row["generate_response"]
    reference_answer = row["target"]
    score = bleu.sentence_score(
        hypothesis=generated_response, references=[reference_answer]
    ).score
    return score


tqdm.pandas()
MT_df = df[df["category"] == "MT"]
MT_df["MT_accuracy"] = MT_df.progress_apply(compute_bleu_score, axis=1)
MT_df.to_json(f"{output_folder}/{output_prefix}_MT.jsonl", lines=True, orient="records")
