import json
import argparse
from tqdm import tqdm
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set_theme(style="whitegrid")

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

def checkLen(data):
    for d in data:
        if len(d)!=10:
            print('a')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="meta", type=str)
    args = parser.parse_args()

    data_dir = "LLM_penetration/results/specific_score"

    if args.dataset_type == 'meta':
        Human_meta = read_data(os.path.join(data_dir, '2019_meta_Human_specific_score.json'))

        GPT4o_meta = read_data(os.path.join(data_dir, '2019_meta_GPT4o_specific_score.json'))
        Gemini_meta = read_data(os.path.join(data_dir, '2019_meta_Gemini_specific_score.json'))
        Claude_meta = read_data(os.path.join(data_dir, '2019_meta_Claude_specific_score.json'))
    
    

        final_data = Human_meta + GPT4o_meta + Gemini_meta + Claude_meta
        for d in final_data:
            d['type'] = args.dataset_type
        # checkLen(Human_meta)
        # checkLen(GPT4o_meta)
        # checkLen(Gemini_meta)
        # checkLen(Claude_meta)
        features = {
        "Sim": "MetaReviewSim", 
        'SFIRF': "MetaReview_SFIRF",    
        
        }

        features_value = {
        "Sim": {"max": 0.7, "min": 0.4}, 
        "SFIRF": {"max": 0.8, "min": 0.2}, 
        
    }
    elif args.dataset_type == 'review':
        Human_meta = read_data(os.path.join(data_dir, 'others_review_Human_specific_score.json'))

        LLM_meta = read_data(os.path.join(data_dir, 'others_review_LLM_specific_score.json'))
        final_data = Human_meta + LLM_meta
        for d in final_data:
            d['type'] = args.dataset_type
        checkLen(Human_meta)
        checkLen(LLM_meta)
        features = {
        "Sim": "ReviewSim", 
        # "ReviewScore_P": "ReviewScore_P",
        # "ReviewScore_R": "ReviewScore_R",
        # "ReviewScore_F1": "ReviewScore_F1",
        # "Review_SF": "Review_SF", 
        # "Review_IRF": "Review_IRF", 
        'SFIRF': "Review_SFIRF",    
        }
        features_value = {
        "Sim": {"max": 1.1, "min": 0.6}, 
        "SFIRF": {"max": 0.2, "min": 0.05}, 
        
    }

        
    
    
    df = pd.DataFrame(final_data)
    cm = plt.get_cmap('Set3')
    
    

    # features_value = {
    #     "AWL": {"max": 7.5, "min": 4.5}, 
    #     "ASL": {"max": 28.0, "min": 15.0}, 
    #     "SWR": {"max": 0.40, "min": 0.2},
    #     "LWR": {"max": 0.25, "min": 0.05},
    #     'FRE': {"max": 50.0, "min": -15.0},
    #     'LSR': {"max": 0.8, "min": 0.2},
        
    #     "TTR": {"max": 0.8, "min": 0.4},
    #     "DRV": {"max": 33.0, "min": 20.0},
    #     "PS": {"max": 0.16, "min": 0.06},
    #     "SS": {"max": 0.50, "min": 0.30},
    #     "SCD": {"max": 1.6, "min": 0.6},
        
    # }
    n_rows, n_cols = 1, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 3))
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5, 3))
    for idx, feature in enumerate(features.keys()):
        print(idx)
        row, col = divmod(idx, n_cols)
        ax = axes[col]
        

        df_filtered = df
        
        # plt.figure(figsize=(5.5, 5.5))
        
        sns.barplot(
            data=df_filtered,
            x="type", y=features[feature], hue="Writer", palette=cm.colors, alpha=.9, hatch='//', 
            # order=['Abstract', 'Meta-Review', 'Review'],
            edgecolor='black', ax=ax)
        

        
        ax.set_ylim(bottom=features_value[feature]['min'], top=features_value[feature]['max'])
        if feature == 'Sim':
            ax.set_title('MetaReview: MRSim')
            ax.set_ylabel('MRSim')
        else:
            ax.set_title('MetaReview: SFIRF')
            ax.set_ylabel('SFIRF')
        ax.set_xlabel('')
        # ax.tick_params(axis='x', labelsize=13)
        ax.set_xticks([])
        ax.tick_params(axis='y')

    for ax in axes.flat[:7]:
        ax.legend_.remove()
    fig_width = plt.gcf().get_size_inches()[0]
    legend_width = fig_width / 2
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
    handles, labels,
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=11
)


    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=2, frameon=False,
    # handlelength=legend_width)
    plt.tight_layout(h_pad=2,w_pad=2)
    plt.savefig(f'LLM_penetration/figure/output/{args.dataset_type}_specific_compare.pdf', dpi=300, bbox_inches="tight")

    print('a')





if __name__ == "__main__":
    main()
