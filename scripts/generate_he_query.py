import seaborn as sns
import matplotlib.pyplot as plt
import copy
import glob
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from evaluate import load

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from datasets import load_dataset
humaneval_dataset = load_dataset("openai_humaneval")["test"]


def estimate_duplicates(samples, num_selection, repeats=20, deduplicates=False):
    import random
    duplicates = []
    for _ in range(repeats):
        sampled_samples = random.sample(samples, num_selection)
        if deduplicates:
            duplicates.append(len(set(sampled_samples)))
        else:
            duplicates.append(len(sampled_samples) - len(set(sampled_samples)))
    return sum(duplicates) / len(duplicates)



def get_reference(doc):
    test_func = doc["test"]
    entry_point = f"check({doc['entry_point']})"
    return "\n" + test_func + "\n" + entry_point

refs = [get_reference(data) for data in humaneval_dataset]

code_metric = load(
    "/scratch/08401/ywen/code/metrics/code_eval",
)


def evaluate_generation(gen_data, references):
    results, _ = code_metric.compute(
        references=references,
        predictions=gen_data,
        k=[1,5,20],
        num_workers=32,
        raw=True
    )
    return results


def find_true_indices(bool_list):
    return [index for index, value in enumerate(bool_list) if value]


def subset_from_indices(original_list, indices_set):
    # Use list comprehension to select elements from the original list at the given indices
    subset = [original_list[i] for i in indices_set if i < len(original_list)]
    return subset


generation_path = Path("/scratch/08401/ywen/code/latest_code_eval/bigcode-evaluation-harness/results/he/humaneval_magicoderpeftft_codellama7b_bs192_lorar16_cosinelr2e-4_run2_ckpt900_temp0.8.json")
high_temp_generation_data = json.load(generation_path.open("r"))
high_temp_results = evaluate_generation(high_temp_generation_data, refs)
generation_path = Path("/scratch/08401/ywen/code/latest_code_eval/bigcode-evaluation-harness/results/he/humaneval_magicoderpeftft_codellama7b_bs192_lorar16_cosinelr2e-4_run2_ckpt900_temp0.1.json")
low_temp_generation_data = json.load(generation_path.open("r"))
low_temp_results = evaluate_generation(low_temp_generation_data, refs)

low_temp_atleastone = [any(res) for res in low_temp_results["raw"]]
high_temp_atleastone = [any(res) for res in high_temp_results["raw"]]
interested_problem_idx = [i for i, (h_leastone, l_leastone) in enumerate(zip(high_temp_atleastone, low_temp_atleastone)) if not l_leastone and h_leastone]
#true_indices = [find_true_indices(res) for res in high_temp_results["raw"]]
correct_solutions = [list(set(subset_from_indices(gen, find_true_indices(res))))
                     for gen, res in zip(high_temp_generation_data, high_temp_results["raw"])]
correct_solutions = [list(map(lambda x: x.strip().replace(humaneval_dataset[idx]["prompt"].strip(), ""), sol_list))
                     for idx, sol_list in enumerate(correct_solutions)]

interested_dataset = humaneval_dataset.add_column("llm_solutions", correct_solutions)
interested_dataset = interested_dataset.select(interested_problem_idx)


def count_solution_length(example):
    example["solution_lengths"] = len(example["llm_solutions"])
    return example

interested_dataset = interested_dataset.map(count_solution_length)

dataset_save_name = "interested_he"
with Path(f"{dataset_save_name}.jsonl").open("w") as f:
    f.writelines(map(lambda data_point: json.dumps(data_point) + '\n', interested_dataset))

solution_lengths = []
problem_indices = [1, 7, 8, 34, 43]
solution_indices = [[0, 2], [0, 3], [0, 15], [0, 2], [0, 4]]
import pdb; pdb.set_trace()