# 按年读取并在每年跑结果
# 用三个种子的三个模型推理abstract 和 meta_review的结果即可
import argparse
import os
from sklearn import metrics
import json
import numpy as np
import data_helper
from transformers import LongformerForSequenceClassification, LongformerTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset
import torch
from safetensors.torch import load_file
from collections import defaultdict
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',default=r"../save_models",type=str)
parser.add_argument('--flag',default=0,type=int) # 默认为推理模型
parser.add_argument('--type',default='abstract_base',type=str) # 训练模型
parser.add_argument('--model_path',default='/data/LLM_Model/longformer-large-4096',type=str)
parser.add_argument('--rawdata_file',default="../data/ICLR_2017_2019_pol_seg.json",type=str)
parser.add_argument('--file',default = '../data/ICLR_2020_2024_key.json',type=str)
parser.add_argument('--review_file',default = '../data/NLP_review.json',type=str) # 不需要review
parser.add_argument('--journal',default='ijcai',type=str) # 期刊名称
parser.add_argument('--label_nums',default=2,type=int)
parser.add_argument('--count',default=1,type=int)
parser.add_argument('--test_batch_size',default=64,type=int)
parser.add_argument('--write_dir',default='../outputs',type=str)
args = parser.parse_args()

def cd_model_path(model_dir,type,LLM):
    path = os.path.join(model_dir,type+'_'+LLM)
    path = os.path.join(path,os.listdir(path)[0])
    path = os.path.join(path,'model.safetensors')
    return path

def data_loader(file):
    with open(file,'r') as f:
        data = json.load(f)
        # data = data[:8] # 读取测试条目
    # 返回年份的dict
    res = {}
    # res = defaultdict(lambda: {'abstract': [], 'meta_review': []})
    for d in data:
        if d['year'] not in res:
            res.setdefault(d['year'], {'abstract': [], 'meta_review': []})
        res[d['year']]['abstract'].append(data_helper.cleaned_text(d['abstract']))
        res[d['year']]['meta_review'].append(data_helper.cleaned_text(d['meta-review']))

    return res

def data_loader_review(file):
    with open(file,'r') as f:
        data = json.load(f)
        # data = data[:8] # 读取测试条目
    res = {}
    for d in data:
        if d['year'] not in res:
            res.setdefault(d['year'], {'review': []})
        for i in range(1,7): # key -- review1-6
            if f'review{i}' in d:
                res[d['year']]['review'].append(data_helper.cleaned_text(d[f'review{i}'])) # 添加review
    return res

def _data_loader(file):
    with open(file,'r') as f:
        data = json.load(f)
        # data = data[:8] # 读取测试条目
    res = {}
    for d in data:
        if str(d['year']) not in res:
            res.setdefault(str(d['year']), {'abstract': []})
        res[str(d['year'])]['abstract'].append(data_helper.cleaned_text(d['abstract']))
    return res


def model_eval(model_name,eval_texts,year,type):
    batch = args.test_batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        tokenizer = LongformerTokenizer.from_pretrained(args.model_path)  # 从本地加载 tokenizer
        if type == 'meta_ex':
            print('here')
            print('*'*20)
            model = LongformerForSequenceClassification.from_pretrained(args.model_path, num_labels=3,torch_dtype=torch.float16)
        elif type != 'abstract':
            model = LongformerForSequenceClassification.from_pretrained(args.model_path, num_labels=args.label_nums)
        else:
            model = LongformerForSequenceClassification.from_pretrained(args.model_path, num_labels=4) # 否则四分类
        state_dict = load_file(model_name) # 读取模型
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        prediction = torch.tensor([],dtype=torch.long).to(device)
        with torch.no_grad():
            for i in range(0,len(eval_texts), batch):
                inputs = tokenizer(eval_texts[i:i+batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, axis=1)
                prediction = torch.cat((prediction,preds))
        prediction = prediction.cpu().numpy()
        # 计算1的个数
        if type == 'meta_ex':
            # 细分两种percentage
            per_1 = str(np.sum([1 for p in prediction if int(p) == 1])*100/len(prediction)).replace('.', '_') # pol
            per_2 = str(np.sum([1 for p in prediction if int(p) == 2])*100/len(prediction)).replace('.', '_') # gen
            save_data = {
            'journal': args.journal.upper(),
            'year':year,
            'numbers': len(prediction), # 该期刊年度的推理条数
            'model':model_name.split('/')[-3],
            'type':type,
            'pol_per':per_1,
            'gen_per':per_2,
            'count':args.count
            }
            # 保存npy的结果
            write_result = os.path.join(args.write_dir,'meta_ex')
            # if not os.path.exist(write_result):
            #     os.mkdir(write_result)

            name = model_name.split('/')[-3].split('_')[0]
            write_result = os.path.join(write_result,f'{name}_{type}_{year}_{args.count}.npy')
            np.save(write_result,prediction)
        else:
            percentage = str(np.sum(prediction)*100 / len(prediction)).replace('.', '_') # 便于复制粘贴
            # 推理的数据只有百分比
            save_data = {
            'journal': args.journal.upper(),
            'year':year,
            'numbers': len(prediction), # 该期刊年度的推理条数
            'model':model_name.split('/')[-3],
            'type':type,
            'percentage':percentage,
            'count':args.count
            }
    else:
        save_data = {'cpu':'empty'}
    
    if args.journal.upper() != 'ICLR':
    
        print('writing the prediction result for the journals')
        write_file = os.path.join(args.write_dir, 'detect_eval_journals.json')
    else:
        if type == 'meta_ex':
            print('writing the prediction_ex result for the journals')
            write_file = os.path.join(args.write_dir, 'meta_ex.json')
        else:
            print('writing the prediction result for ICLR')
            write_file = os.path.join(args.write_dir, 'detect_eval_ICLR.json')
    with open(write_file,'a') as f:
        f.seek(0, 2)  # 移动到文件末尾
        if f.tell() == 0:
            # 如果文件为空，直接写入数据
            f.write(json.dumps(save_data, indent=4,ensure_ascii=False))
        else:
            # 否则，加逗号分隔新的字典
            f.write(',\n')
            f.write(json.dumps(save_data, indent=4,ensure_ascii=False))

if __name__ == '__main__':
    # 用最好的模型进行推理 -- ICLR + 其他刊
    if args.count == 1:
        model_dir = os.path.join(args.save_dir, '_1')
    elif args.count == 2:
        model_dir = os.path.join(args.save_dir,'_2')
    elif args.count == 3:
        model_dir = os.path.join(args.save_dir,'_3')
    
    if args.journal.upper() == 'ICLR':
        data_file = '../data/ICLR_2020_2024_key.json'
        if args.type != 'review':
            model_path = cd_model_path(model_dir,'hybrid','mix') # 指定hybrid的mix模型进行推理
            data = data_loader(data_file)

        else:
            model_path = cd_model_path(model_dir,'meta_review','mix') # 指定meta_review的mix模型推理review
            data = data_loader_review(data_file)

    else:
        model_path = cd_model_path(model_dir,'hybrid','mix') # 指定hybrid的mix模型进行推理 -- 其他journal的abstract
        data_file = f'../data/journal/{args.journal.upper()}_2020-2024_abstract.json'
        data = _data_loader(data_file)
    if args.journal.upper() != 'ICLR' and args.count != 1:
        print('The other journals have been alreaady evaled.')
        # 不继续推理
    # else:

    # 额外处理
    if args.type == 'meta_ex':
        model_path = cd_model_path(model_dir,'meta_ex','gpt')
        data = data_loader(data_file)
    

    for i in range(2020,2025):
        year = str(i)
        try:
            eval = data[year] # 指定年份
        except:
            continue # 没有这个年份的数据则跳过
        if args.type != 'review': # 不是review
            abstract = eval['abstract'] # 每个刊都要推理abstract
            # model_eval(model_path,abstract,year,'abstact') # 进入推理，返回百分比
            if args.journal.upper() == 'ICLR': # 只有ICLR刊需要推理meta-review
                meta_review = eval['meta_review']
                model_eval(model_path,meta_review,year,'meta_ex') 
        else:
            review = eval['review']
            model_eval(model_path,review,year,'review') # 推理review

    print(f'finish eval -- {args.type} -- {args.journal.upper()}')