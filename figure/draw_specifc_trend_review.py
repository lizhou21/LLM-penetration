import json
import argparse
from tqdm import tqdm
import spacy
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
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

def checkLen(data,year):
    for d in data:
        if len(d)!=10:
            print('a')
            print(year)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="review", type=str)
    args = parser.parse_args()
    data_dir = "LLM_penetration/results/specific_score"

    final_data = []
    if args.dataset_type == 'meta':
        for year in ['2019', '2020', '2021', '2022', '2023', 2024]:
            Human_meta = read_data(os.path.join(data_dir, f'{year}_meta_Human_specific_score.json'))
            final_data = final_data + Human_meta

        features = {
            "Sim": "MetaReviewSim", 
            # "MetaReviewScore_P": "MetaReviewScore_P",
            # "MetaReviewScore_R": "MetaReviewScore_R",
            # "MetaReviewScore_F1": "MetaReviewScore_F1",
            # "MetaReview_SF": "MetaReview_SF", 
            # "MetaReview_IRF": "MetaReview_IRF", 
            'SFIRF': "MetaReview_SFIRF",    
        }

    elif args.dataset_type == 'review':
        for year in ['2019', '2020', '2021', '2022', '2023', '2024']:
            print(year)
            Human_meta = read_data(os.path.join(data_dir, f'{year}_review_Human_specific_score.json'))
            final_data = final_data + Human_meta
        
            checkLen(Human_meta,year)

        features = {
        "Sim": "ReviewSim", 
        # "ReviewScore_P": "ReviewScore_P",
        # "ReviewScore_R": "ReviewScore_R",
        # "ReviewScore_F1": "ReviewScore_F1",
        # "Review_SF": "Review_SF", 
        # "Review_IRF": "Review_IRF", 
        'SFIRF': "Review_SFIRF",    
        
    }

    
    
    df = pd.DataFrame(final_data)
    cm = plt.get_cmap('Set2')
   
    

    
    # for i in ['Abstract', 'Meta-Review', 'Review']:
    n_rows, n_cols = 1, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 3))
    for idx, feature in enumerate(features.keys()):
        
        print(idx)
        row, col = divmod(idx, n_cols)
        ax = axes[col]


        df_filtered = df
        
        df_filtered = df_filtered[df_filtered['Writer'] == 'Human']
        # df_filtered = df_filtered[df_filtered['type'] == i]
        # plt.figure(figsize=(5.5, 5.5))
        desired_order=['≤2019', '2020', '2021', '2022', '2023', '2024']
        df_filtered["year"] = pd.Categorical(df_filtered["Year"], categories=desired_order, ordered=True)
        
        sns.lineplot(
            data=df_filtered,
            x="Year", y=features[feature], palette=cm.colors, ax=ax)
        

        
        # ax.set_ylim(bottom=features_value[feature]['min'], top=features_value[feature]['max'])
        # ax.set_title(features[feature], fontsize=17)
        # ax.set_ylabel(feature,fontsize=17)
        ax.set_title(features[feature])
        ax.set_ylabel(feature)
        if feature == 'Sim':
            ax.set_title('Review: RSim')
            ax.set_ylabel('RSim')
        else:
            ax.set_title('Review: SFIRF')
            ax.set_ylabel('SFIRF')
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=10)
        # ax.tick_params(axis='y', labelsize=13)
        # ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        for label in ax.get_xticklabels():
                label.set_rotation(45)

    # for ax in axes.flat:
    #     ax.legend_.remove()
    fig_width = plt.gcf().get_size_inches()[0]
    legend_width = fig_width / 2
    # handles, labels = axes[0, 0].get_legend_handles_labels()
    # fig.legend(
    #     handles, labels,
    #     loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False, fontsize=20
    # )


    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=2, frameon=False,
    # handlelength=legend_width)
    plt.tight_layout()
    plt.savefig(f'LLM_penetration/figure/output/specific_trend_{args.dataset_type}.png', dpi=300, bbox_inches="tight")

    print('a')





if __name__ == "__main__":
    main()
