import copy
import json
import pathlib
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import torch.nn.functional as F

import transformers
from transformers import Trainer, DefaultDataCollator
from datasets import load_dataset
from torch_influence import LiSSAInfluenceModule, BaseObjective, IdentityInfluenceModule


PROMPT_TEMPLATE = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction} Your code should satisfy the following assertion:
```python
{assertation}
```

@@ Response
```python
"""


def transfer_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (tuple, list)):
        return type(batch)(transfer_to_device(x, device) for x in batch)
    elif isinstance(batch, dict):
        return {k: transfer_to_device(x, device) for k, x in batch.items()}
    else:
        raise NotImplementedError()


class MyObjective(BaseObjective):
    def __init__(
        self,
        model,
        trainer,
    ):
        self.model = model
        self.trainer = trainer

    def train_outputs(self, model, batch):
        if isinstance(batch, dict) and not isinstance(batch["input_ids"], torch.Tensor):
            data_collator = DefaultDataCollator()
            batch = data_collator([batch])
            batch = transfer_to_device(batch, model.device)
        outputs = model(**batch) 
        return outputs["logits"]

    def train_loss_on_outputs(self, outputs, batch):
        if isinstance(batch, dict) and not isinstance(batch["input_ids"], torch.Tensor):
            data_collator = DefaultDataCollator()
            batch = data_collator([batch])
            batch = transfer_to_device(batch, self.model.device)
        import pdb; pdb.set_trace()
        return F.cross_entropy(outputs, batch[1])  # mean reduction required

    def train_regularization(self, params):
        return 0.
        #return 0.01 * torch.square(params.norm())

    # training loss by default taken to be 
    # train_loss_on_outputs + train_regularization

    def test_loss(self, model, params, batch):
        if isinstance(batch, dict) and not isinstance(batch["input_ids"], torch.Tensor):
            data_collator = DefaultDataCollator()
            batch = data_collator([batch])
            batch = transfer_to_device(batch, model.device)
        return self.trainer.compute_loss(self.model, batch)

    def train_loss(self, model, params, batch):
        if isinstance(batch, dict) and not isinstance(batch["input_ids"], torch.Tensor):
            data_collator = DefaultDataCollator()
            batch = data_collator([batch])
            batch = transfer_to_device(batch, model.device)
        return self.trainer.compute_loss(self.model, batch)


IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"
PAD_TOKEN = "<|PAD|>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="codellama/CodeLlama-7b-Python-hf")
    peft_model: Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank."},
    )
    problem_index: int = field(
        default=12,
        metadata={"help": "which query example to be used."},
    )
    query_idx: int = field(
        default=0,
        metadata={"help": "which query example to be used."},
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    sources = examples["problem"]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['solution']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def test_tokenize_function(examples, tokenizer):
    sources = [PROMPT_TEMPLATE.format(
        instruction=ins, assertation=ass) for ins, ass in zip(examples["instruction"], examples["assertation"])]
    targets = [f"{ex}\n{EOT_TOKEN}" for ex in examples["solution"]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def filter_python_solutions(examples):
    solutions = []
    description = []
    lengths = []
    tests = []
    for sol,  desc, diff, tt in zip(examples["solutions"], examples["description"], examples["difficulty"], examples["public_tests"]):
        current_solution = []
        for lang, candidate_solution in zip(sol["language"], sol["solution"]):
            if lang == 3:
                current_solution.append(candidate_solution)
        if diff < 8 and current_solution:
            solutions.append(current_solution)
            description.append(desc)
            lengths.append(len(current_solution))
            tests.append(tt)
    return {"problem": description, "python_solutions": solutions, "solution_lengths": lengths, "assertation": tests}


def build_problem_testset(examples):
    sol_length = len(examples["python_solutions"][0])
    data_dict = {
        "instruction": sol_length * examples["problem"],
        "solution": examples["python_solutions"][0],
        "assertation": sol_length * examples["assertation"],
    }
    return data_dict


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank == 0:
        print('='*100)
        print(training_args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    # Set pad token in llama
    if not hasattr(tokenizer, "pad_token") or not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if training_args.local_rank == 0:
        print("Load tokenizer from {} over.".format(model_args.model_name_or_path))

    raw_train_datasets = load_dataset(
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        split="train"
    )

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=False,
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )

    raw_test_dataset = load_dataset("deepmind/code_contests")["test"]
    test_dataset = raw_test_dataset.map(
        filter_python_solutions,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_test_dataset.column_names,
        load_from_cache_file=False,
        desc="Running Encoding",
    )
    problem_dataset = test_dataset.select([training_args.problem_index]).map(
        build_problem_testset,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=test_dataset.column_names,
        load_from_cache_file=False,
        desc="Running Encoding",
    )
    test_dataset = problem_dataset.map(
        test_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=problem_dataset.column_names,
        load_from_cache_file=False,
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )

    if model_args.peft_model:
        from peft import PeftModel  # dynamic import to avoid dependency on peft

        model.enable_input_require_grads()
        model = PeftModel.from_pretrained(model, model_args.peft_model, is_trainable=True)
    else:
        from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=training_args.lora_r, lora_alpha=32, lora_dropout=0.1
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


    if training_args.local_rank == 0:
        print(f"index {training_args.query_idx} of the test set: {test_dataset[training_args.query_idx]['input_ids']}, {test_dataset[training_args.query_idx]['labels']}.")
        print(f"index {training_args.query_idx} of the test set: {tokenizer.decode(list(test_dataset[training_args.query_idx]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    #train_loader = trainer.get_train_dataloader()
    #test_loader = trainer.get_test_dataloader(test_dataset=test_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    #trainer.train()
    #trainer.save_state()
    #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    cache_name=f"influence_cache/codecontest_testquery_{training_args.query_idx}_cache.json"
    module = IdentityInfluenceModule(
        model=model,
        objective=MyObjective(model, trainer),
        train_loader=train_loader,
        test_loader=test_loader,
        device=torch.device("cuda"),
        cache_name=cache_name,
    )

    # influence scores of training points 1, 2, and 3 on test point 0
    #import time
    #start_time = time.time()
    #scores = module.influences(list(range(110)), [training_args.query_idx])
    scores = module.influences(list(range(len(train_dataset))), [training_args.query_idx])
    #print(f"It takes {time.time()-start_time} secs to calculate the scores")
    #print(scores)
    json.dump(
        scores.tolist(),
        pathlib.Path(f"influence_cache/{pathlib.Path(cache_name).stem}_final.json").open('w'),
        indent=2
    )


if __name__ == "__main__":
    main()
