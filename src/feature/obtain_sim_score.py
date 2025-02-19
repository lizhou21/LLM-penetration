

import json
import argparse
# from DetectMetrics import DetectMetrics, Review_SEG
from tqdm import tqdm
import spacy
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
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

def calculate_f1(precision, recall):
    if precision + recall == 0:  # 避免除以零的情况
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="review", type=str)
    parser.add_argument("--year", default="2019", type=str)
    parser.add_argument("--review_sim", default="intra", type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()

    if args.dataset_type == 'review':
        if args.year == '2019':
            file = 'LLM_penetration/results/embedding/2019_review_save_embedding.pkl'
            with open(file, 'rb') as f:
                data_sim = pickle.load(f)
        elif args.year in ['LLM', 'Human']:
            file = 'LLM_penetration/results/embedding/review_LLM_embedding.pkl'
            with open(file, 'rb') as f:
                data_sim = pickle.load(f)
                data_sim = data_sim[args.year]

        else:
            file = 'LLM_penetration/results/embedding/2020_2024_review_save_embedding.pkl'
            with open(file, 'rb') as f:
                data_sim = pickle.load(f)

                data_sim = data_sim[args.year]
                
        

        if args.review_sim == 'intra':
            intra_data = []
            all_paper_sent_SF_IRF_value = []
            all_paper_SF_IRF_value = []
            
            for d in tqdm(data_sim):
                paper_sent_SF_IRF_value = []
                paper_SF_IRF_value = [] # r层面求平均 SF-IRF(s, r, R)  
                all_review_rep = d['reviews_sent_rep']
                for current_review in all_review_rep:
                    current_review_sent_sim = np.vstack(cosine_similarity(current_review)) # 同一条review中，不同sent与其他sent的相似度
                    current_review_sent_sim[current_review_sent_sim < args.threshold] = 0
                    SF_value = np.log(current_review_sent_sim.shape[0]/np.sum(current_review_sent_sim, axis=0))
                    # SF_value = np.mean(current_review_sent_sim, axis=1)  # 当前review的每个句子的SF
                    IRF_max_sim = []
                    for compare_review in all_review_rep:
                        compare_review_sent_sim = np.vstack(cosine_similarity(current_review, compare_review))
                        compare_review_sent_sim[compare_review_sent_sim < args.threshold] = 0
                        max_sim = np.max(compare_review_sent_sim, axis=1)
                        IRF_max_sim.append(max_sim)
                    IRF_value = np.log(len(all_review_rep)/np.sum(np.vstack(IRF_max_sim), axis=0))
                    SF_IR_value = SF_value*IRF_value # SF-IRF(s, r, R) 
                    paper_sent_SF_IRF_value.append(SF_IR_value)
                    paper_SF_IRF_value.append(np.mean(SF_IR_value))
                all_paper_SF_IRF_value.append(np.mean(paper_SF_IRF_value))#取最大的？
                all_paper_sent_SF_IRF_value.append(paper_sent_SF_IRF_value)
                intra_data.append(
                    {
                        'year': args.year,
                        'SF_IRF': float(np.mean(paper_SF_IRF_value)), 
                    }
                )
            wirte_data(intra_data, f"LLM_penetration/results/sim_score/{args.year}_{args.dataset_type}_{args.review_sim}.json")
            print(f'{args.year}: {np.mean(all_paper_SF_IRF_value)}')
            print('a')

        elif args.review_sim == 'ReviewScore':
            intra_data = []
            all_paper_P_score = []
            all_paper_R_score = []
            all_paper_F1_score = []
            
            for d in tqdm(data_sim):
                paper_P_score = []
                paper_R_score = []
                paper_F1_score = []
                all_review_rep = d['reviews_sent_rep']
                for i, current_review in enumerate(all_review_rep):
                    for j, compare_review in enumerate(all_review_rep):
                        if i!=j:
                            review_sent_sim = np.vstack(cosine_similarity(current_review, compare_review)) # 同一条review中，不同sent与其他sent的相似度
                            R_score = np.sum(np.max(review_sent_sim, axis=0))/review_sent_sim.shape[0]
                            P_score = np.sum(np.max(review_sent_sim, axis=1))/review_sent_sim.shape[1]
                            F1_score = calculate_f1(P_score, R_score)
                            paper_P_score.append(P_score)
                            paper_R_score.append(R_score)
                            paper_F1_score.append(F1_score)
                all_paper_P_score.append(np.max(paper_P_score))
                all_paper_R_score.append(np.max(paper_R_score))
                all_paper_F1_score.append(np.max(paper_F1_score))

                intra_data.append(
                    {
                        'year': args.year,
                        'p': float(np.max(paper_P_score)),
                        'r': float(np.max(paper_R_score)),
                        'f1': float(np.max(paper_F1_score)), 
                    }
                )
            wirte_data(intra_data, f"LLM_penetration/results/sim_score/{args.year}_{args.dataset_type}_{args.review_sim}.json")
            print(f'{args.year}: p-{np.mean(all_paper_P_score)}, r-{np.mean(all_paper_R_score)}, f1-{np.mean(all_paper_F1_score)}')
            print('a')



if __name__ == "__main__":
    main()