# 读取并构建数据
import json
import random
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
import re
import random
import argparse
import os


def cleaned_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:()-]', '', text)  # 使用正则表达式去除非文本符号（保留字母、数字、空格和标点符号）
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # 可选：移除多余的空格
    # cleaned_text = cleaned_text.replace('\n','') # 删去换行符
    return cleaned_text


def data2txt(texts, labels, data_name, type, LLMs):
    write_dir = os.path.join('../data', data_name)  # 手动定义txt_file
    if not os.path.exists(write_dir):  # 创建文件夹
        os.makedirs(write_dir)
    if type == 'train':
        with open(os.path.join(write_dir, f'{type}_{LLMs}.txt'), 'w') as f:
            for text, label in zip(texts, labels):
                f.write(f'{text}\t{label}\n')
        print(f'the {type}_{LLMs} data has been written in {data_name}')
    else:
        if os.path.exists(os.path.join(write_dir, f'{type}')):  # 如果测试集已经写入就不重复写入
            print(f'the {type} data has already been written in {data_name}')
            return None
        with open(os.path.join(write_dir, f'{type}.txt'), 'w') as f:
            for text, label in zip(texts, labels):
                f.write(f'{text}\t{label}\n')
        print(f'the {type} data has been written in {data_name}')


def to_meta_check(file, LLMs='gpt'):  # 不用写入txt
    with open(file, 'r') as f:
        dataset = json.load(f)
    f.close()
    # 划分数据集
    train, test = split_data(dataset)

    def loader(data, cata, LLMs):
        labels, texts = [], []
        claude_meta = []
        for d in data:
            claude_meta.append(d['meta_review_claude'])

        for d in data:

            # LLM_text = cleaned_text(d['meta_review_gpt'])
            human_text = cleaned_text(d['meta_review']['comment'])

            if LLMs == 'claude':  # 分两种情况
                LLM_text = cleaned_text(d['meta_review_claude'])
                labels.append(1)
                texts.append(LLM_text)
                labels.append(0)
                texts.append(human_text)
            else:
                raise 'the LLMs source is not a available check source, "claude" only !'

        return Dataset.from_dict({'text': texts, 'label': labels})

    train_dataset, test_dataset = loader(train, LLMs), loader(test, LLMs)
    return train_dataset, test_dataset


def to_abstract_base(file, LLMs='gpt'):  # 默认gpt
    with open(file, 'r') as f:
        dataset = json.load(f)
    f.close()
    train, test = split_data(dataset)

    def loader(data, type, LLMs):
        labels, texts = [], []
        if type == 'train':
            for d in data:
                abstract = cleaned_text(d['abstract'])
                if LLMs == 'gpt':
                    polish_abstract = cleaned_text(d['polish_abstract'])
                elif LLMs == 'gemini':
                    polish_abstract = cleaned_text(d['polish_abstract_gemini'])
                elif LLMs == 'claude':
                    polish_abstract = cleaned_text(d['polish_abstract_claude'])
                elif LLMs == 'mix':
                    polish_abstract = cleaned_text(d['abstract_mix'])
                if abstract and (polish_abstract):
                    labels.append(0)
                    texts.append(abstract)
                    labels.append(1)
                    texts.append(polish_abstract)
            data2txt(texts, labels, 'abstract', type, LLMs)
        else:
            for d in data:
                abstract = cleaned_text(d['abstract'])
                texts.append(abstract)
                labels.append(0)
                texts.extend([cleaned_text(d['polish_abstract']), cleaned_text(d['polish_abstract_gemini']),
                              cleaned_text(d['polish_abstract_claude'])])
                labels.extend([1, 1, 1])
            data2txt(texts, labels, 'abstract', type, LLMs)
        return Dataset.from_dict({'text': texts, 'label': labels})

    train_dataset, test_dataset = loader(train, 'train', LLMs), loader(test, 'test', LLMs)
    return train_dataset, test_dataset


def to_meta_ex(file, LLMs='gpt'):  # for the compensate experiment -- only for the mix source

    with open(file, 'r') as f:
        dataset = json.load(f)
    f.close()
    train, dev, test = split_data(dataset)

    def loader(data, type, LLMs):
        labels, texts = [], []
        if type == 'train':
            for d in data:
                # human -- 0 pol -- 1 gen -- 2
                # LLM_text = cleaned_text(d['meta_review_gpt'])
                human_text = cleaned_text(d['meta_review']['comment'])
                gen_text = cleaned_text(d['meta_review_gpt'])
                if d['polish_meta_review_gpt'] != None:
                    pol_text = cleaned_text(d['polish_meta_review_gpt'])
                    labels.append(0)
                    texts.append(human_text)
                    labels.append(1)
                    texts.append(pol_text)
                    labels.append(2)
                    texts.append(gen_text)

            data2txt(texts, labels, 'meta_ex', type, LLMs)
        else:  # 添加gpt生成的gen和pol 包括meta和abstract
            for d in data:  # 只需要gpt的数据作为测试集
                if d['id'] == 'S1g_EsActm':
                    a = d['id']
                if d['polish_meta_review_gpt'] != None:
                    human_text = cleaned_text(d['meta_review']['comment'])
                    texts.append(human_text)
                    labels.append(0)
                    texts.extend([cleaned_text(d['meta_review_gpt'])])
                    texts.extend([cleaned_text(d['polish_meta_review_gpt'])])
                    labels.extend([2, 1])
                    texts.extend([cleaned_text(d['abstract']), cleaned_text(d['polish_abstract'])])
                    labels.extend([0, 1])  # 添加human和polish
            data2txt(texts, labels, 'meta_ex', type, LLMs)
        return Dataset.from_dict({'text': texts, 'label': labels})

    train_dataset, dev_dataset, test_dataset = loader(train, 'train', LLMs), loader(test, 'dev', LLMs), loader(test,
                                                                                                               'test',
                                                                                                               LLMs)
    return train_dataset, dev_dataset, test_dataset


def to_meta(file, LLMs='gpt'):
    with open(file, 'r') as f:
        dataset = json.load(f)
    f.close()
    # 划分数据集
    train, test = split_data(dataset)

    def loader(data, type, LLMs):
        labels, texts = [], []
        if type == 'train':
            for d in data:

                # LLM_text = cleaned_text(d['meta_review_gpt'])
                human_text = cleaned_text(d['meta_review']['comment'])
                if LLMs == 'gpt':
                    LLM_text = cleaned_text(d['meta_review_gpt'])
                elif LLMs == 'gemini':
                    LLM_text = cleaned_text(d['meta_review_gemini'])
                elif LLMs == 'claude':
                    LLM_text = cleaned_text(d['meta_review_claude'])
                elif LLMs == 'mix':
                    LLM_text = cleaned_text(d['meta_mix'])
                if LLM_text and human_text:
                    labels.append(1)
                    texts.append(LLM_text)
                    labels.append(0)
                    texts.append(human_text)
            data2txt(texts, labels, 'meta', type, LLMs)
        else:
            for d in data:
                human_text = cleaned_text(d['meta_review']['comment'])
                texts.append(human_text)
                labels.append(0)
                texts.extend([cleaned_text(d['meta_review_gpt']), cleaned_text(d['meta_review_gemini']),
                              cleaned_text(d['meta_review_claude'])])
                labels.extend([1, 1, 1])
            data2txt(texts, labels, 'meta', type, LLMs)
        return Dataset.from_dict({'text': texts, 'label': labels})

    train_dataset, test_dataset = loader(train, 'train', LLMs), loader(test, 'test', LLMs)
    return train_dataset, test_dataset


def to_abstract(file):
    with open(file, 'r') as f:
        dataset = json.load(f)
    f.close()
    train, test = split_data(dataset)

    def loader(data, type):
        labels, texts = [], []

        for d in data:
            abstract = cleaned_text(d['abstract'])
            polish_abstract = cleaned_text(d['polish_abstract'])
            generate = cleaned_text(d['human_generate'])
            generate_polish = cleaned_text(d['human_generate_polish'])

            if abstract and (polish_abstract and generate and generate_polish):
                labels.append(0)
                texts.append(abstract)
                labels.append(1)
                texts.append(polish_abstract)
                labels.append(2)
                texts.append(generate)
                labels.append(3)
                texts.append(generate_polish)
        return Dataset.from_dict({'text': texts, 'label': labels})

    train_dataset, test_dataset = loader(train, 'train'), loader(test, ' test')
    return train_dataset, test_dataset


def to_hybrid(file, LLMs):
    # 测试集不用管,只需要训练集混合即可
    with open(file, 'r') as f:
        dataset = json.load(f)
    f.close()
    train, test = split_data(dataset)

    def loader(data, type, LLMs):
        labels, texts = [], []
        if type == 'train':
            for d in data:
                abstract = cleaned_text(d['abstract'])
                if LLMs == 'gpt':
                    polish_abstract = cleaned_text(d['polish_abstract'])
                elif LLMs == 'gemini':
                    polish_abstract = cleaned_text(d['polish_abstract_gemini'])
                elif LLMs == 'claude':
                    polish_abstract = cleaned_text(d['polish_abstract_claude'])
                elif LLMs == 'mix':
                    polish_abstract = cleaned_text(d['abstract_mix'])

                human_text = cleaned_text(d['meta_review']['comment'])
                if LLMs == 'gpt':
                    LLM_text = cleaned_text(d['meta_review_gpt'])
                elif LLMs == 'gemini':
                    LLM_text = cleaned_text(d['meta_review_gemini'])
                elif LLMs == 'claude':
                    LLM_text = cleaned_text(d['meta_review_claude'])
                elif LLMs == 'mix':
                    LLM_text = cleaned_text(d['meta_mix'])
                if (LLM_text and human_text) and (abstract and (polish_abstract)):
                    labels.append(0)
                    texts.append(abstract)
                    labels.append(1)
                    texts.append(polish_abstract)
                    # labels.append(1)
                    # texts.append(generate)
                    # labels.append(1)
                    # texts.append(generate_polish)
                    labels.append(1)
                    texts.append(LLM_text)
                    labels.append(0)
                    texts.append(human_text)
            data2txt(texts, labels, 'hybird', type, LLMs)
        # else:
        #     for d in data:
        #         abstract = cleaned_text(d['abstract'])
        #         texts.append(abstract)
        #         labels.append(0)
        #         texts.extend([d['polish_abstract'],d['polish_abstract_gemini'],d['polish_abstract_claude']])
        #         labels.extend([1,1,1])
        #     data2txt(texts, labels, type, LLMs)
        return Dataset.from_dict({'text': texts, 'label': labels})

    train_dataset = loader(train, 'train', LLMs)
    return train_dataset  # 不返回tester


def split_data(data_pairs):
    total_pairs = len(data_pairs)
    train_size = 0.7
    dev_size = 0.2
    train_end = int(train_size * total_pairs)
    dev_end = int(dev_size * total_pairs) + train_end
    train_pairs = data_pairs[:train_end]
    dev_pairs = data_pairs[train_end:dev_end]
    test_pairs = data_pairs[train_end:]

    return train_pairs, dev_pairs, test_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='../data/ICLR_2017_2019_pol_seg_ex.json', type=str)
    parser.add_argument('--txt_dir', default='../data', type=str)
    args = parser.parse_args()
    LLMss = ['claude', 'gpt']
    # for LLM in LLMss:
    # train_dataset, test_dataset = to_abstract_base(args.file,LLMs=LLM)
    # train_dataset, test_dataset = to_meta(args.file,LLMs=LLM)
    # train_dataset = to_hybrid(args.file,LLMs=LLM)
    # trainset,testset = to_meta_ex(args.file,LLMs='gpt')
    trainset, testset = to_meta_ex(args.file, LLMs='gpt')



