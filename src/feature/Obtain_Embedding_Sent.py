import json
import argparse
from DetectMetrics import DetectMetrics, Review_SEG
from tqdm import tqdm
import spacy
from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="review", type=str)
    parser.add_argument("--year", default="others", type=str)
    parser.add_argument("--embedding_type", default="sent_embedding", type=str)
    args = parser.parse_args()
    print(f"{args.year}: {args.dataset_type}")
    dataset_dir = "LLM_penetration/data"
    save_dir = "LLM_penetration/results"

    model = SentenceTransformer("/mntcephfs/data/haizhouli/LLM-models/all-MiniLM-L6-v2")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU 或 CPU
    model = model.to(device)

    nlp = spacy.load("en_core_web_sm")


    if args.year == "2019":
        dataset_file = os.path.join(dataset_dir, 'ICLR_2017_2019_pol_seg.json')
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if args.dataset_type == 'review':
            save_data = []
            for d in tqdm(data):
                save_d = {}
                article_embeddings = []
                articles = []
                for r in d['reviews']:
                    review = Review_SEG(r['review'],nlp)
                    articles.append(review.sentences)
                    embeddings = model.encode(review.sentences, device=device)
                    article_embeddings.append(embeddings)
                save_d['reviews_sent'] = articles
                save_d['reviews_sent_rep'] = article_embeddings
                save_data.append(save_d)
            with open(os.path.join(save_dir, args.embedding_type, f'{args.year}_{args.dataset_type}_{args.embedding_type}.pkl'), 'wb') as f:
                pickle.dump(save_data, f)
        
        elif args.dataset_type == 'meta':
            save_data = {
            'Human': [],
            'GPT4o': [],
            'Gemini': [],
            'Claude': [],
        }
            for d in tqdm(data):
                save_d = {}

                human_articles = Review_SEG(d['meta_reivew']["comment"], nlp).sentences
                human_embeddings = model.encode(human_articles, device=device)
                save_data['Human'].append(
                    {
                        'reviews_sent': human_articles,
                        'reviews_sent_rep': human_embeddings
                    }
                )

                gpt_articles = Review_SEG(d['meta_review_gpt'], nlp).sentences
                gpt_embeddings = model.encode(gpt_articles, device=device)
                save_data['GPT4o'].append(
                    {
                        'reviews_sent': gpt_articles,
                        'reviews_sent_rep': gpt_embeddings
                    }
                )

                gemini_articles = Review_SEG(d['meta_review_gemini'], nlp).sentences
                gemini_embeddings = model.encode(gemini_articles, device=device)
                save_data['Gemini'].append(
                    {
                        'reviews_sent': gemini_articles,
                        'reviews_sent_rep': gemini_embeddings
                    }
                )


                claude_articles = Review_SEG(d['meta_review_claude'], nlp).sentences
                claude_embeddings = model.encode(claude_articles, device=device)
                save_data['Claude'].append(
                    {
                        'reviews_sent': claude_articles,
                        'reviews_sent_rep': claude_embeddings
                    }
                )


            with open(os.path.join(save_dir, args.embedding_type, f'{args.year}_{args.dataset_type}_{args.embedding_type}.pkl'), 'wb') as f:
                pickle.dump(save_data, f)
        
    elif args.year == "2024":
        dataset_file = os.path.join(dataset_dir, 'ICLR_2020_2024_key.json')
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if args.dataset_type == 'review':
            save_data = {
                '2020': [],
                '2021': [],
                '2022': [],
                '2023': [],
                '2024': []
                }
            for d in tqdm(data):
                save_d = {}
                articles = []
                for key, value in d.items():
                    if key not in ['year', 'abstract', 'meta-review']:
                        articles.append(Review_SEG(value,nlp).sentences)
                save_d['reviews_sent'] = articles
                article_embeddings = []
                
                for r in articles:
                    embeddings = model.encode(r, device=device)
                    article_embeddings.append(embeddings)
                save_d['reviews_sent_rep'] = article_embeddings
                year = d['year']
                save_data[year].append(save_d)
            
            with open(os.path.join(save_dir, args.embedding_type, f'{args.year}_{args.dataset_type}_{args.embedding_type}.pkl'), 'wb') as f:
                pickle.dump(save_data, f)
        elif args.dataset_type == 'meta':
            save_data = {
                '2020': [],
                '2021': [],
                '2022': [],
                '2023': [],
                '2024': []
                }
            for d in tqdm(data):
                save_d = {}
                for key, value in d.items():
                    if key in ['meta-review']:
                        articles = Review_SEG(value,nlp).sentences
                save_d['reviews_sent'] = articles
                article_embeddings = model.encode(articles, device=device)
                save_d['reviews_sent_rep'] = article_embeddings
                year = d['year']
                save_data[year].append(save_d)
            with open(os.path.join(save_dir, args.embedding_type, f'{args.year}_{args.dataset_type}_{args.embedding_type}.pkl'), 'wb') as f:
                pickle.dump(save_data, f)


    
    elif args.year == 'others':
        dataset_file = os.path.join(dataset_dir, 'review_LLM_human.json')
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if args.dataset_type == 'review':

            save_data = {
                'Human': [],
                'GPT4o': [],
                'Gemini': [],
                'Claude': [],
            }
            for d in tqdm(data):
                save_d = {}
                article_embeddings = []
                articles = []

                for r in d['review_human']:
                    review = Review_SEG(r['review'],nlp)
                    human_articles = review.sentences
                    human_embeddings = model.encode(human_articles, device=device)
                    articles.append(human_articles)
                    article_embeddings.append(human_embeddings)

                    save_data['Human'].append(
                        {
                            'reviews_sent': articles,
                            'reviews_sent_rep': article_embeddings
                        }
                    )

                gpt_articles = Review_SEG(d['review_gpt'], nlp).sentences
                gpt_embeddings = model.encode(gpt_articles, device=device)
                save_data['GPT4o'].append(
                    {
                        'reviews_sent': gpt_articles,
                        'reviews_sent_rep': gpt_embeddings
                    }
                )

                gemini_articles = Review_SEG(d['review_gemini'], nlp).sentences
                gemini_embeddings = model.encode(gemini_articles, device=device)
                save_data['Gemini'].append(
                    {
                        'reviews_sent': gemini_articles,
                        'reviews_sent_rep': gemini_embeddings
                    }
                )


                claude_articles = Review_SEG(d['review_claude'], nlp).sentences
                claude_embeddings = model.encode(claude_articles, device=device)
                save_data['Claude'].append(
                    {
                        'reviews_sent': claude_articles,
                        'reviews_sent_rep': claude_embeddings
                    }
                )


            with open(os.path.join(save_dir, args.embedding_type, f'{args.year}_{args.dataset_type}_{args.embedding_type}.pkl'), 'wb') as f:
                pickle.dump(save_data, f)

if __name__ == "__main__":
    main()