import json
import argparse
from DetectMetrics import DetectMetrics
from tqdm import tqdm
import spacy
import os

def read_data(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        ret = []
        for i, item in enumerate(f.readlines()):
            record = json.loads(item)
            ret.append(record)
    return ret

def wirte_data(list, file_path):
    
    with open(file_path, 'w', encoding='utf-8') as  f:
        for l in list:
            json_str = json.dumps(l, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')

def main():
    print('begin')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="abstract", type=str)
    parser.add_argument("--year", default="2020", type=str)
    args = parser.parse_args()

    data_dir = 'LLM_penetration/data'
    save_dir = 'LLM_penetration/results/general_score'

    nlp = spacy.load("en_core_web_sm")

    if args.year == '2019':
        dataset_file = os.path.join(data_dir, 'ICLR_2017_2019_pol_seg.json')
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data_score = []
        for d in tqdm(data):
            ## 1. abstract
            abstract_human = DetectMetrics(d['abstract'], nlp, 0).get_metrics()
            abstract_human['type'] = 'Abstract'
            abstract_human['writer'] = 'Human'
            abstract_human['year']="≤2019"
            data_score.append(abstract_human)

            abstract_gpt4o = DetectMetrics(d['polish_abstract'], nlp, 0).get_metrics()
            abstract_gpt4o['type'] = 'Abstract'
            abstract_gpt4o['writer'] = 'GPT4o'
            abstract_gpt4o['year']="≤2019"
            data_score.append(abstract_gpt4o)

            abstract_gemini = DetectMetrics(d['polish_abstract_gemini'], nlp, 0).get_metrics()
            abstract_gemini['type'] = 'Abstract'
            abstract_gemini['writer'] = 'Gemini'
            abstract_gemini['year']="≤2019"
            data_score.append(abstract_gemini)

            abstract_claude = DetectMetrics(d['polish_abstract_claude'], nlp, 0).get_metrics()
            abstract_claude['type'] = 'Abstract'
            abstract_claude['writer'] = 'Claude'
            abstract_claude['year']="≤2019"
            data_score.append(abstract_claude)

            ## 2. review
            for r in d['reviews']:
                review_human = DetectMetrics(r['review'], nlp, 1).get_metrics()
                review_human['type'] = 'Review'
                review_human['writer'] = 'Human'
                review_human['year']="≤2019"
                data_score.append(review_human)

            ## 3. meta-review
            meta_human = DetectMetrics(d['meta_reivew']['comment'], nlp, 0).get_metrics()
            meta_human['type'] = 'Meta-Review'
            meta_human['writer'] = 'Human'
            meta_human['year']="≤2019"
            data_score.append(meta_human)

            meta_gpt4o = DetectMetrics(d['meta_review_gpt'], nlp, 0).get_metrics()
            meta_gpt4o['type'] = 'Meta-Review'
            meta_gpt4o['writer'] = 'GPT4o'
            meta_gpt4o['year']="≤2019"
            data_score.append(meta_gpt4o)

            meta_gemini = DetectMetrics(d['meta_review_gemini'], nlp, 0).get_metrics()
            meta_gemini['type'] = 'Meta-Review'
            meta_gemini['writer'] = 'Gemini'
            meta_gemini['year']="≤2019"
            data_score.append(meta_gemini)

            meta_claude = DetectMetrics(d['meta_review_claude'], nlp, 0).get_metrics()
            meta_claude['type'] = 'Meta-Review'
            meta_claude['writer'] = 'Claude'
            meta_claude['year']="≤2019"
            data_score.append(meta_claude)

            wirte_data(data_score, os.path.join(save_dir, f'{args.year}_all_general_score.json'))

    elif args.year == '2024':
        dataset_file = os.path.join(data_dir, 'ICLR_2020_2024.json')
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_score = []
        for d in tqdm(data):
            for key, value in d.items():
                if key == 'year':
                    year = value
                else:
                    if key == 'abstract':
                        d_score = DetectMetrics(value, nlp, 0).get_metrics()
                        d_score['type'] = 'Abstract'
                        d_score['year'] = year
                        d_score['writer'] = 'Human'
                    elif key == 'meta-review':
                        d_score = DetectMetrics(value, nlp, 0).get_metrics()
                        d_score['type'] = 'Meta-Review'
                        d_score['year'] = year
                        d_score['writer'] = 'Human'
                    else:
                        d_score = DetectMetrics(value, nlp, 0).get_metrics()
                        d_score['type'] = 'Review'
                        d_score['year'] = year
                        d_score['writer'] = 'Human'

                    data_score.append(d_score)
        wirte_data(data_score, os.path.join(save_dir, f'{args.year}_all_general_score.json'))
    
    elif args.year == 'others':
        dataset_file = os.path.join(data_dir, 'review_LLM_human.json')
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_score = []
        for d in tqdm(data):
            ## 1. abstract
            ## 2. review
            for r in d['review_human']:
                review_human = DetectMetrics(r['review'], nlp, 1).get_metrics()
                review_human['type'] = 'Review'
                review_human['writer'] = 'Human'
                review_human['year']="≤2019"
                data_score.append(review_human)

            ## 3. meta-review

            review_gpt4o = DetectMetrics(d['review_gpt'], nlp, 1).get_metrics()
            review_gpt4o['type'] = 'Review'
            review_gpt4o['writer'] = 'GPT4o'
            review_gpt4o['year']="≤2019"
            data_score.append(review_gpt4o)

            review_gemini = DetectMetrics(d['review_gemini'], nlp, 1).get_metrics()
            review_gemini['type'] = 'Review'
            review_gemini['writer'] = 'Gemini'
            review_gemini['year']="≤2019"
            data_score.append(review_gemini)

            review_claude = DetectMetrics(d['review_claude'], nlp, 1).get_metrics()
            review_claude['type'] = 'Review'
            review_claude['writer'] = 'Claude'
            review_claude['year']="≤2019"
            data_score.append(review_claude)

            
        wirte_data(data_score, os.path.join(save_dir, f'LLM_review_general_score.json'))




if __name__ == "__main__":
    main()