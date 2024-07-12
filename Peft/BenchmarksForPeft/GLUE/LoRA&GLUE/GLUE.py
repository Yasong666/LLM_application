from args import get_args
import os
import time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import(
    get_peft_model,
    get_peft_config,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
)
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm

## hyperparameters
# arg
arg_seed  = 40
arg_task = "sst2"
arg_peft_type = "lora"
arg_device = "cuda"
arg_num_labels = 2
arg_model_name_or_path = "bert-base-uncased"
arg_bs = 16
arg_fft_lr = 1e-6
arg_weight_decay = 0.01
arg_head_lr = 0.01
arg_num_epochs = 100

torch.manual_seed(arg_seed)#设置随机数种子
peft_type = arg_peft_type
task = arg_task
device = arg_device
num_labels = arg_num_labels
peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
if any(k in arg_model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"
tokenizer = AutoTokenizer.from_pretrained(arg_model_name_or_path, padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("glue", task)
metric = load_metric("glue", task)

def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    if task == 'sst2' or task == 'cola':
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=args.max_length)
    elif task == 'qnli':
        outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=args.max_length)
    elif task == 'qqp':
        outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=args.max_length)
    else:
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=args.max_length)
    return outputs

if task == 'sst2' or task == 'cola':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence"],
    )
elif task == 'qnli':
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "question", "sentence"],
    )
elif task == 'qqp':
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "question1", "question2"],
    )
else:
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
    )
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")
# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=arg_bs)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=arg_bs
)

model = AutoModelForSequenceClassification.from_pretrained(arg_model_name_or_path,num_labels=num_labels,return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model

head_param = list(map(id, model.classifier.parameters()))

others_param = filter(lambda p: id(p) not in head_param, model.parameters()) 

optimizer = AdamW([
    {"params": model.classifier.parameters(), "lr": arg_head_lr},
    {"params": others_param, "lr": arg_fft_lr}
],weight_decay=arg_weight_decay)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * arg_num_epochs),
    num_training_steps=(len(train_dataloader) * arg_num_epochs),
)

acc_list = []
model.to(device)

###train阶段

for epoch in range(arg_num_epochs):
    model.train()
    for step,batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    model.eval()
    for step,batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        if task == 'stsb':
            predictions = outputs.logits
        else:
            predictions = outputs.logits.argmax(-1)
        predictions, refernces = predictions, batch["labels"]
        metric.add_batch(predictions, refernces)

    eval_metric = metric.compute()
    if task == "stsb":
        acc_list.append(eval_metric['pearson'])
        # log(f"epoch {epoch}:", eval_metric, ', current_best_pearson:',max(acc_list),'train_loss:',loss)
        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_pearson:\033[0m',max(acc_list),'train_loss:',loss)
    elif task == 'cola':
        acc_list.append(eval_metric['matthews_correlation'])
        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_corr:\033[0m',max(acc_list),'train_loss:',loss)
        # log(f"epoch {epoch}:", eval_metric, ', current_best_corr:',max(acc_list),'train_loss:',loss)
    else:
        acc_list.append(eval_metric['accuracy'])
        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_acc:\033[0m',max(acc_list),'train_loss:',loss)
        # log(f"epoch {epoch}:", eval_metric, ', current_best_acc:',max(acc_list),'train_loss:',loss)


##突然发现，GPT2一般是用来做文本生成的，不需要用这个来做NLU，嗨~感觉是有点浪费时间了