

import json
import argparse
from DetectMetrics import *
from tqdm import tqdm
import spacy
import numpy as np
import torch
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="review", type=str)
    parser.add_argument("--year", default="others", type=str)
    parser.add_argument("--writer", default="LLM", type=str)
    args = parser.parse_args()
    num = {
        '2019': 2831,
        '2020': 2213, 
        '2021': 2594, 
        '2022': 2619,
        '2023': 3797,
        '2024': 5780
    }

    
    data_dir = 'LLM_penetration/results'
    all_data = []
    if args.year == '2019':
        if args.dataset_type =='meta':
            for embed_type in ['para_embedding', 'sent_embedding']:
                meta_file = os.path.join(data_dir, embed_type, f'{args.year}_meta_{embed_type}.pkl')
                review_file = os.path.join(data_dir, embed_type, f'{args.year}_review_{embed_type}.pkl')
                with open(meta_file, 'rb') as f:
                    Meta_Rep = pickle.load(f)
                with open(review_file, 'rb') as f:
                    Review_Rep = pickle.load(f)
          
                # for writer in ['Human', 'GPT4o', 'Gemini', 'Claude']:
                Meta_wirter = Meta_Rep[args.writer]
                for i, each_meta_rep in enumerate(tqdm(Meta_wirter)):
                    single_data = {
                        'Year': '≤2019',
                        'Writer': args.writer,
                        # 'Embed_Type': embed_type,
                    }
                    curret_meta_rep = each_meta_rep['reviews_sent_rep']
                    if embed_type == 'para_embedding':
                        meta_review_sim = MetaReviewSim(curret_meta_rep, Review_Rep[i]['reviews_sent_rep'])
                        single_data['MetaReviewSim'] = float(meta_review_sim.meta_review_sim)
                        all_data.append(single_data)
                        
                    elif embed_type == 'sent_embedding':
                        SFIRF_socre = MetaReviewSFIRF(curret_meta_rep, Review_Rep[i]['reviews_sent_rep'])
                        all_data[i]['MetaReview_SFIRF'] = float(SFIRF_socre.SFIRF['SF_IRF'])
                        
    
        elif args.dataset_type =='review':
            for embed_type in ['para_embedding', 'sent_embedding']:
                review_file = os.path.join(data_dir, embed_type, f'{args.year}_review_{embed_type}.pkl')
                with open(review_file, 'rb') as f:
                    Review_Rep = pickle.load(f)

                for i, reviews_info in enumerate(Review_Rep):
                    all_reviews = Review_Rep[i]['reviews_sent_rep']
                    single_data = {
                        'Year': '≤2019',
                        'Writer': 'Human',
                    }
                    if embed_type == 'para_embedding':
                        review_sim = ReviewSim(all_reviews)
                        single_data['ReviewSim'] = float(review_sim.ReviewSim)
                        all_data.append(single_data)
                        
                    elif embed_type == 'sent_embedding':
                        SFIRF_socre = ReviewSFIRF(all_reviews)
                        all_data[i]['Review_SF'] = float(SFIRF_socre.SFIRF['SF'])
                        all_data[i]['Review_IRF'] = float(SFIRF_socre.SFIRF['IRF'])
                        all_data[i]['Review_SFIRF'] = float(SFIRF_socre.SFIRF['SF_IRF'])
                    
    
    elif args.year == 'others':
        for embed_type in ['para_embedding', 'sent_embedding']:
            review_file = os.path.join(data_dir, embed_type, f'{args.year}_review_{embed_type}.pkl')
            with open(review_file, 'rb') as f:
                Review_Rep_ALL = pickle.load(f)
            
            Review_Rep_Human = Review_Rep_ALL['Human']
            Review_Rep_GPT4o = Review_Rep_ALL['GPT4o']
            Review_Rep_Gemini = Review_Rep_ALL['Gemini']
            Review_Rep_Claude = Review_Rep_ALL['Claude']
            # Review_Rep = [Review_Rep_GPT4o, Review_Rep_Gemini, Review_Rep_Claude]


            for i, reviews_info in enumerate(tqdm(Review_Rep_GPT4o)):
                GPT4o = Review_Rep_GPT4o[i]['reviews_sent_rep']
                Gemini = Review_Rep_Gemini[i]['reviews_sent_rep']
                Claude = Review_Rep_Claude[i]['reviews_sent_rep']
                if args.writer == 'LLM':
                    all_reviews = [GPT4o, Gemini, Claude]
                else:
                    all_reviews=Review_Rep_Human[i]['reviews_sent_rep']
                single_data = {
                    'Year': '≤2019',
                    'Writer': args.writer,
                }
                if embed_type == 'para_embedding':
                    review_sim = ReviewSim(all_reviews)
                    single_data['ReviewSim'] = float(review_sim.ReviewSim)
                    all_data.append(single_data)
                    
                elif embed_type == 'sent_embedding':
                    SFIRF_socre = ReviewSFIRF(all_reviews)
                    all_data[i]['Review_SF'] = float(SFIRF_socre.SFIRF['SF'])
                    all_data[i]['Review_IRF'] = float(SFIRF_socre.SFIRF['IRF'])
                    all_data[i]['Review_SFIRF'] = float(SFIRF_socre.SFIRF['SF_IRF'])

        print('a')
    
    else:
        if args.dataset_type =='meta':
            for embed_type in ['para_embedding', 'sent_embedding']:
                meta_file = os.path.join(data_dir, embed_type, f'2024_meta_{embed_type}.pkl')
                review_file = os.path.join(data_dir, embed_type, f'2024_review_{embed_type}.pkl')
                with open(meta_file, 'rb') as f:
                    Meta_Rep_ALL = pickle.load(f)
                with open(review_file, 'rb') as f:
                    Review_Rep_ALL = pickle.load(f)
                # for year, Meta_Rep in Meta_Rep_ALL.items():
                Meta_Rep = Meta_Rep_ALL[args.year]
                Review_Rep = Review_Rep_ALL[args.year]

                for i, each_meta_rep in enumerate(tqdm(Meta_Rep)):
                    single_data = {
                        'Year': args.year,
                        'Writer': args.writer,
                    }
                    curret_meta_rep = each_meta_rep['reviews_sent_rep']
                    if embed_type == 'para_embedding':
                        meta_review_sim = MetaReviewSim(curret_meta_rep, Review_Rep[i]['reviews_sent_rep'])
                        single_data['MetaReviewSim'] = float(meta_review_sim.meta_review_sim)
                        all_data.append(single_data)
                        
                    elif embed_type == 'sent_embedding':
                        SFIRF_socre = MetaReviewSFIRF(curret_meta_rep, Review_Rep[i]['reviews_sent_rep'])
                        all_data[i]['MetaReview_SFIRF'] = float(SFIRF_socre.SFIRF['SF_IRF'])


        elif args.dataset_type =='review':
            for embed_type in ['para_embedding', 'sent_embedding']:
                review_file = os.path.join(data_dir, embed_type, f'2024_review_{embed_type}.pkl')
                with open(review_file, 'rb') as f:
                    Review_Rep_ALL = pickle.load(f)
                Review_Rep = Review_Rep_ALL[args.year]
                for i, reviews_info in enumerate(tqdm(Review_Rep)):
                    all_reviews = Review_Rep[i]['reviews_sent_rep']
                    single_data = {
                        'Year': args.year,
                        'Writer': 'Human',
                    }
                    if embed_type == 'para_embedding':
                        review_sim = ReviewSim(all_reviews)
                        single_data['ReviewSim'] = float(review_sim.ReviewSim)
                        all_data.append(single_data)
                        
                    elif embed_type == 'sent_embedding':
                        SFIRF_socre = ReviewSFIRF(all_reviews)
                        all_data[i]['Review_SF'] = float(SFIRF_socre.SFIRF['SF'])
                        all_data[i]['Review_IRF'] = float(SFIRF_socre.SFIRF['IRF'])
                        all_data[i]['Review_SFIRF'] = float(SFIRF_socre.SFIRF['SF_IRF'])


    save_dir = 'LLM_penetration/results/specific_score'
    save_file = f'{save_dir}/{args.year}_{args.dataset_type}_{args.writer}_specific_score.json'
    wirte_data(all_data, save_file)

                        


    


if __name__ == "__main__":
    main()