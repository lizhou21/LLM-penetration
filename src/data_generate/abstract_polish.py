from datetime import timedelta

import openai
from langchain_community.chat_models import ChatOpenAI
import warnings
import argparse
import asyncio
import pandas as pd
from polish_prompt import polish_prompt
import re
import json
import os
import time
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default="./result", type=str)
parser.add_argument("--model", default="gemini-1.5-pro-002", type=str)
# gpt-4o-2024-08-06； claude-3-opus-20240229； gemini-1.5-pro-002
parser.add_argument("--data_dir", default="../data/abstract_data", type=str)
parser.add_argument("--read_file", default="../generate_result/ICLR_2017_2019_pol_seg_ex.json", type=str)
parser.add_argument("--write_file", default="../generate_result/ICLR_2017_2019_pol_seg_ex.json", type=str)
args = parser.parse_args()


openai.api_base = ""
openai.api_key = ""


async def gpt_generate(abstract):
    count = 0
    while count < 5:
        try:
            prompt_list = polish_prompt(abstract)
            # 调用random随机取整数 -- 随机取一个prompt
            index = random.randint(0, len(prompt_list) - 1)
            prompt = prompt_list[index]
            response = await openai.ChatCompletion.acreate(
                model=args.model,
                messages=[
                    {'role': 'user', 'content': prompt}],
                temperature=0.9,
            )

            # 获取内容并清理
            res = response['choices'][0]['message']['content'].replace('\n', '').replace('\t', '')
            return res
        except Exception as e:
            # 捕获错误并打印详细信息
            print(f"Error in API call for prompt: {abstract}, Error: {e}")
            # 报错不返回值
            if count < 5:
                count += 1
                print('第{}次尝试'.format(count))
            else:
                return abstract

# 改成读取json -- 从data_dir中载入
async def process_in_batches(read_dir=args.data_dir, read_file='ICLR_2017-2019.json', batch_size=10):
    # 按照批次处理
    results = []
    with open(os.path.join(read_dir, read_file), 'r') as f:
        paper_data = json.load(f)
    f.close()
    # paper_data = paper_data[:10]
    nums = len(paper_data)




    for i in tqdm(range(0, nums, batch_size)):
        # batch_prompts = [df.iloc[j]['src'] for j in range(i, min(i + batch_size, df.shape[0]))]
        batch_abstracts = [paper_data[j] for j in range(i,min(i+batch_size,nums))]
        batch_abstract = [batch_abstracts[j%batch_size]['src'] for j in range(i,min(i+batch_size,nums))]
        tasks = [gpt_generate(abstract) for abstract in batch_abstract] # 创建任务
        batch_results = await asyncio.gather(*tasks, return_exceptions=True) # 执行任务
        # 记录时间
        # print(get_time_dif(start_time=start_time))
        results.extend(batch_results)  # 存放结果

    return results # 所有2017（测试）的润色摘要


async def main():
    results = await process_in_batches()  # 每次处理10条
    return results


# 现在处理results这个列表，合并到总paper的polish中
# ICLR_2017-2019.json
def add_key(results,read_file=args.read_file):
    with open(read_file, 'r') as f: # 先读原始数据
        paper_data = json.load(f)
    f.close()
    with open(args.write_file, 'w') as wf:
        # 再写结果数据
        write_result = []
        for i,data in enumerate(paper_data):
            data_new = {}
            for k in data.keys():
                if k == 'gemi_segments':
                    continue
                data_new[k] = data[k]
                # 插入到指定的后面
                if k == 'polish_abstract_gemini':
                    data_new['polish_abstract_claude'] = results[i]
                if k == 'gemi_segments':
                    pass
            write_result.append(data_new)
        # 把list写入json
        json.dump(write_result, wf,ensure_ascii=False,indent=4)
    wf.close()

def get_time_dif(start_time):
    end_time = time.time()
    diff = end_time - start_time
    return timedelta(seconds=int(round(diff)))


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())
    add_key(result)
