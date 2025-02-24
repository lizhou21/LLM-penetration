import os
import random
from datetime import datetime
from transformers import LongformerForSequenceClassification, LongformerTokenizer, TrainingArguments, Trainer, \
    AutoModelForSequenceClassification
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from datasets import load_dataset, Dataset
from safetensors.torch import load_file
from transformers import TrainerCallback
import json
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn import metrics
import argparse
from deployment import detect
from transformers import pipeline
import warnings
import pdb
from data_helper import *
from transformers import AutoModel, AutoConfig, AutoTokenizer

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default=r"../save_models", type=str)
parser.add_argument('--flag', default=0, type=int)  # 默认为推理模型
parser.add_argument('--type', default='meta_review', type=str)  # 训练模型
parser.add_argument('--model_path', default='allenai/longformer-large-4096', type=str)
parser.add_argument('--rawdata_file', default="../data/ICLR_2017_2019_pol_seg.json", type=str)
parser.add_argument('--file', default='../data/ICLR_2017_2019_pol_seg_ex.json', type=str)
parser.add_argument('--review_file', default='../data/NLP_review.json', type=str)
parser.add_argument('--label_nums', default=2, type=int)
parser.add_argument('--count', default=1, type=int)
parser.add_argument('--test_batch_size', default=64, type=int)
parser.add_argument('--write_file', default='../outputs/claude_check.json', type=str)
parser.add_argument('--write_file_base', default='../outputs/detect_test_base.json', type=str)
parser.add_argument('--LLM', default='gpt', type=str)
args = parser.parse_args()


def cleaned_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:()-]', '', text)  # 使用正则表达式去除非文本符号（保留字母、数字、空格和标点符号）
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # 可选：移除多余的空格
    return cleaned_text


def load_test_data(data_file, type):
    with open(data_file, 'r') as f:
        data = json.load(f)
    f.close()
    if type == 'review':
        data_test = data
    else:
        train_end = int(0.7 * len(data))
        dev_end = train_end + int(0.2 * len(data))  # 与训练时候的代码构造一致
        data_test = data[dev_end:]
    if type == 'abstract':
        # 构造text label对
        texts, labels = data_pairs_abstract(data_test)
    if type == 'meta_review':
        texts, labels = data_pais_metareview(data_test)
    if type == 'review':
        texts, labels = data_pairs_review(data_test)

    return texts, labels


def data_pairs_abstract(pairs):
    texts = []
    labels = []
    for pair in pairs:
        if pair['abstract']:
            texts.append(pair['abstract'])
            labels.append(0)
        if pair['polish_abstract']:
            texts.append(pair['polish_abstract'])
            labels.append(1)
        if pair['human_generate']:
            texts.append(pair['human_generate'])
            labels.append(1)
        if pair['human_generate_polish']:
            texts.append('human_generate_polish')
            labels.append(1)
    return texts, labels


def data_pais_metareview(pairs):
    texts = []
    labels = []
    for pair in pairs:
        if pair['result']:
            texts.append(pair['result'])
            labels.append(1)
        if pair['meta-review']:
            texts.append(pair['meta-review'])
            labels.append(0)
    return texts, labels


def data_pairs_review(pairs):
    texts = []
    labels = []
    for pair in pairs:
        if pair['type'] == 'human':
            for r in pair['reviews']:
                r = cleaned_text(r)
                texts.append(r)
                labels.append(0)
        else:
            for r in pair['reviews']:
                r = cleaned_text(r)
                texts.append(r)
                labels.append(1)
    return texts, labels


def data_loader(type, LLMs=args.LLM):
    if type != 'review':
        if type == 'abstract':
            train_dataset, test_dataset = to_abstract(file=args.file)
        elif type == 'abstract_base':
            train_dataset, test_dataset = to_abstract_base(file=args.file, LLMs=LLMs)
        elif type == 'meta_review':
            train_dataset, test_dataset = to_meta(file=args.file, LLMs=LLMs)
            # 写成两个函数
        elif type == 'meta_ex':
            train_dataset, dev_dataset, test_dataset = to_meta_ex(file=args.file, LLMs=LLMs)
        # elif type=='hybrid': # hybrid没有单独test集
        #     train_dataset, test_dataset = to_hybrid(file=args.file,LLMs=LLMs)
        elif type == 'meta_review_claude_check':
            train_dataset, test_dataset = to_meta_check(file=args.file, LLMs=LLMs)
        texts, labels = test_dataset['text'], test_dataset['label']
    else:
        texts, labels = load_test_data(args.review_file, type=type)
    return texts, labels


def _metrics(labels, prediction):
    F1 = metrics.f1_score(labels, prediction, average='micro')
    F1_per_class = metrics.f1_score(labels, prediction, average=None)
    acc = metrics.accuracy_score(labels, prediction)
    # 一种便于复制粘贴有限小数的若至方法
    return str(F1 * 100).replace('.', '_'), [str(per * 100).replace('.', '_') for per in F1_per_class], str(
        acc * 100).replace(',', '_')


def model_eval(model_name, type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(type)
        tokenizer = LongformerTokenizer.from_pretrained(args.model_path)  # 从本地加载 tokenizer
        if type == 'meta_ex':
            print('here')
            model = LongformerForSequenceClassification.from_pretrained(args.model_path, num_labels=3,
                                                                        torch_dtype=torch.float16)
        elif type != 'abstract':
            model = LongformerForSequenceClassification.from_pretrained(args.model_path, num_labels=args.label_nums,
                                                                        torch_dtype=torch.float16)

        else:
            model = LongformerForSequenceClassification.from_pretrained(args.model_path, num_labels=4)  # 否则四分类
        state_dict = load_file(model_name)  # 读取模型
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    texts, labels = data_loader(type)

    # texts, labels = texts[:10], labels[:10]
    batch = args.test_batch_size
    prediction = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        for i in range(0, len(labels), batch):
            inputs = tokenizer(texts[i:i + batch], padding=True, truncation=True, max_length=512,
                               return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, axis=1)
            prediction = torch.cat((prediction, preds))
    labels = np.array(labels)
    prediction = prediction.cpu().numpy()
    F1, F1_per_class, acc = _metrics(labels, prediction)
    print('{} -- {} -- {} -- The F1 score: {}'.format(args.count, model_name, type, F1), flush=True)
    save_data = {
        'model': model_name.split('/')[-3],
        'type': type,
        'LLM': args.LLM,
        'F1_per': list(F1_per_class),
        'F1': F1,
        'acc': acc,
        'count': args.count
    }
    with open(args.write_file, 'a') as f:
        f.seek(0, 2)  # 移动到文件末尾
        if f.tell() == 0:
            # 如果文件为空，直接写入数据
            f.write(json.dumps(save_data, indent=4, ensure_ascii=False))
        else:
            # 否则，加逗号分隔新的字典
            f.write(',\n')
            f.write(json.dumps(save_data, indent=4, ensure_ascii=False))

    return F1


def pipe_eval(type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    texts, labels = data_loader(type)
    # texts, labels = texts[:2], labels[:2]
    # pipe = pipeline("text-classification", model="roberta-base-openai-detector")
    pipe = pipeline("text-classification", model="MayZhou/e5-small-lora-ai-generated-detector", device=device)
    batch = args.test_batch_size
    prediction = torch.tensor([], dtype=torch.long).to(device)
    for i in range(0, len(labels), batch):
        texts = [text[:512] for text in texts]  # 人工截断
        print(pipe(texts[i:i + batch]))
        # 手动设置0.85的threshold
        pred = torch.tensor([0 if item['score'] < 0.85 else 1 for item in pipe(texts[i:i + batch], )],
                            dtype=torch.long).to(device)  # [{'label': 'Real', 'score': 0.8036582469940186}]
        prediction = torch.cat((prediction, pred))
    prediction = prediction.cpu().numpy()
    labels = np.array(labels)
    F1, F1_per_class, acc = _metrics(labels, prediction)
    print('Pipe -- The F1 score: {}'.format(F1))
    save_data = {
        'model': 'e5-small',
        'type': type,
        'F1': F1,
        'F1_per': list(F1_per_class),
        'acc': acc,
        'count': args.count
    }
    print(save_data)
    with open(args.write_file_base, 'a') as f:
        f.seek(0, 2)  # 移动到文件末尾
        print('open----------------')
        if f.tell() == 0:
            # 如果文件为空，直接写入数据
            f.write(json.dumps(save_data, indent=4, ensure_ascii=False))
        else:
            # 否则，加逗号分隔新的字典
            f.write(',\n')
            f.write(json.dumps(save_data, indent=4, ensure_ascii=False))
    return F1


def model_eval_mage(model_path, type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.to(device)
    # model.load_state_dict(torch.load(model_path)) # 读取模型
    model.eval()
    texts, labels = data_loader(type)
    # texts, labels = texts[:10], labels[:10]
    batch = args.test_batch_size
    prediction = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        for i in range(0, len(labels), batch):
            # text = [preprocess(t) for t in texts[i:i+batch]]
            result = [detect(t, tokenizer, model, device) for t in texts[i:i + batch]]
            preds = [0 if res == 'human-written' else 1 for res in result]
            preds = torch.tensor(preds, dtype=torch.long).to(device)
            prediction = torch.cat((prediction, preds))
            # print(prediction,flush=True)
    labels = np.array(labels)
    prediction = prediction.cpu().numpy()
    prediction = [1 if pred >= 1 else 0 for pred in prediction]  # 把非1的改为1 -- AI
    F1, F1_per_class, acc = _metrics(labels, prediction)
    print('{} -- Model mage -- {} -- The F1 score: {}'.format(args.count, type, F1), flush=True)
    save_data = {
        'model': 'mage',
        'type': type,
        'F1_per': list(F1_per_class),
        'F1': F1,
        'acc': acc,
        'count': args.count
    }
    print(save_data)
    with open(args.write_file_base, 'a') as f:
        f.seek(0, 2)  # 移动到文件末尾
        print('open----------------')
        if f.tell() == 0:
            # 如果文件为空，直接写入数据
            f.write(json.dumps(save_data, indent=4, ensure_ascii=False))
        else:
            # 否则，加逗号分隔新的字典
            f.write(',\n')
            f.write(json.dumps(save_data, indent=4, ensure_ascii=False))
    return F1


def model_eval_roberta(model_path, type):
    texts, labels = data_loader(type)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device)
    prediction = torch.tensor([], dtype=torch.long).to(device)
    for i in range(0, len(texts), args.test_batch_size):
        batch_texts = texts[i:i + args.test_batch_size]
        # Load the saved state dict of the fine-tuned model
        model.load_state_dict(torch.load(model_path), strict=False)
        model.eval()
        inputs = tokenizer(
            batch_texts,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)

        # Get the predicted label using the input_ids and attention_mask
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        preds = torch.argmax(outputs.logits, axis=1)
        prediction = torch.cat((prediction, preds))
    prediction = prediction.cpu().numpy()
    labels = np.array(labels)
    F1, F1_per_class, acc = _metrics(labels, prediction)
    save_data = {
        'model': 'roberta',
        'type': type,
        'F1_per': list(F1_per_class),
        'F1': F1,
        'acc': acc,
        'count': args.count
    }
    with open(args.write_file_base, 'a') as f:
        f.seek(0, 2)  # 移动到文件末尾
        print('open----------------')
        if f.tell() == 0:
            # 如果文件为空，直接写入数据
            f.write(json.dumps(save_data, indent=4, ensure_ascii=False))
        else:
            # 否则，加逗号分隔新的字典
            f.write(',\n')
            f.write(json.dumps(save_data, indent=4, ensure_ascii=False))
    return F1


def cd_model_path(model_dir, type, LLM):
    path = os.path.join(model_dir, type + '_' + LLM)
    path = os.path.join(path, os.listdir(path)[0])
    path = os.path.join(path, 'model.safetensors')
    return path


if __name__ == '__main__':

    # 排列组合，读取model和file并推理得到F1
    if args.count == 1:
        model_dir = os.path.join(args.save_dir, '_1')
    elif args.count == 2:
        model_dir = os.path.join(args.save_dir, '_2')
    elif args.count == 3:
        model_dir = os.path.join(args.save_dir, '_3')

    model_path = [cd_model_path(model_dir, 'abstract_base', args.LLM),
                  cd_model_path(model_dir, 'meta_review', args.LLM), cd_model_path(model_dir, 'hybrid', args.LLM)]

    model_name = ['abstract_base', 'meta_review', 'review']
    # for the test data
    for model in model_path:
        for type in model_name:
            model_eval(model, type)
            print('*' * 20)

    # data_file, data_name = [os.path.join(args.data_dir,'ICLR_2017-2019.json'),os.path.join(args.data_dir,'ICLR_2017_2019_abstract.json'),os.path.join(args.data_dir,'NLP_review.json')], ['meta_review','abstract','review']
    # model_path = './zh_model/LLMDect-classification-longformer.pt'
    # for data, type in zip(data_file,data_name):
    #     model_eval_zh(model_path,data,type)
    #     print('*'*20)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # baseline for MAGE
    model_path = "yaful/MAGE"  # model in the online demo
    if args.LLM == 'gpt':
        for type in model_name:
            model_eval_mage(model_path, type)
            print('*' * 20)
    model_path = '../save_models/best_model.pt'
    # baseline for RAIDetect
    for type in model_name:
        model_eval_roberta(model_path, type)

