import os
import random
from datetime import datetime
from transformers import LongformerForSequenceClassification, LongformerTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from safetensors.torch import load_file
from transformers import TrainerCallback
import json
import torch
from tqdm import tqdm
import pandas as pd
from sklearn import metrics
import argparse
from transformers import pipeline
import re
from data_helper import *
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',default=r"../save_models",type=str)
parser.add_argument('--flag',default=0,type=int) # default = train
parser.add_argument('--type',default='meta_ex',type=str) 
parser.add_argument('--model_path',default='allenai/longformer-large-4096',type=str)
parser.add_argument('--rawdata_file',default="../data/ICLR_2017_2019_pol_seg.json",type=str)
parser.add_argument('--file',default = '../data/ICLR_2017_2019_pol_seg_ex.json',type=str)
parser.add_argument('--label_nums',default=2,type=int)
parser.add_argument('--count',default=2,type=int)
parser.add_argument('--LLMs',default='gpt',type=str) # the LLM source for training
args = parser.parse_args()


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(logs)
    print('train')

def train(train_dataset,dev_dataset):
    training_args = TrainingArguments(
        save_strategy="steps",
        save_steps=1000,
        output_dir=out_dir,
        eval_strategy="steps",
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_steps=1000,
        gradient_accumulation_steps=32,
        fp16=True,
        logging_dir=f"{args.save_dir}/logs",
        logging_steps=100,
        overwrite_output_dir=True,
        seed=seed,
    )



    trainer = Trainer(
        model=model,
        callbacks=[PrinterCallback()],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )


    trainer.train()

    trainer.save_model(out_dir) # 保存到对应模型下
def main():
    if args.type=='abstract':
        train_dataset, test_dataset = to_abstract(file=args.file)
    elif args.type=='abstract_base':
        train_dataset, test_dataset = to_abstract_base(file=args.file,LLMs=args.LLMs)
    elif args.type=='meta_review':
        train_dataset, test_dataset = to_meta(file=args.file,LLMs=args.LLMs)
    elif args.type=='hybrid':
        train_dataset, test_dataset = to_hybrid(file=args.file,LLMs=args.LLMs)
    elif args.type=='meta_ex':
        train_dataset, dev_dataset, test_dataset = to_meta_ex(file=args.file,LLMs=args.LLMs)   

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    dev_dataset = dev_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train(train_dataset,dev_dataset)
    print(args.type + 'finish')

if __name__ == '__main__':
    seed = 235789234 #73583512 #42629309
    random.seed(seed)
    out_dir = os.path.join(args.save_dir,'_' + str(args.count))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, args.type + '_' + args.LLMs)

    print(out_dir)
    if not os.path.exists(out_dir): # makedir
        os.mkdir(out_dir)
    tokenizer = LongformerTokenizer.from_pretrained(args.model_path)  # load tokenizer
    model = LongformerForSequenceClassification.from_pretrained(args.model_path, num_labels=args.label_nums)
    # load dataset
    main()
