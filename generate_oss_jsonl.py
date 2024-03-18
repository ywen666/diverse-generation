import json
from pathlib import Path
from datasets import load_dataset


oss_train_data = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
import random
split_label = random.choices(range(4), k=len(oss_train_data))
oss_train_data = oss_train_data.add_column("split_label", split_label)
import pdb; pdb.set_trace()
with Path("data-clean-decontaminated.jsonl").open("w") as f:
    for data_point in oss_train_data:
        f.write(f'{json.dumps(data_point)}\n')
