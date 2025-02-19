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

def main():
    data_dir = "LLM_penetration/results/buyao/general_score_c_n_s"

    all_2019_file = os.path.join(data_dir, '2019_all_general_score.json')
    all_2024_file = os.path.join(data_dir, '2024_all_general_score.json')
    all_2019_scores = read_data(all_2019_file)
    all_2024_scores = read_data(all_2024_file)
    final_data = all_2019_scores + all_2024_scores
    
    
    df = pd.DataFrame(final_data)
    cm = plt.get_cmap('Set2')
    features = {
        "AWL": "Average Word Length", 
        "LWR": "Long Word Ratio",
        "SWR": "Stopword Ratio",
        # "TTR": "Type Token Ratio",
        # "ASL": "Average Sentence Length", 
        # "DRV": "Dependency Relation Variety", 
        # 'SCD': "Subordinate Clause Density",
        'FRE': "Readability",
        # "PS": "Sentiment Polarity Score", 
        # 'SS': "Sentiment Subjectivity Score",
        
    }
    

    features_value = {
        "AWL": {"max": 7.5, "min": 4.5}, 
        "ASL": {"max": 27.0, "min": 15.0}, 
        "SWR": {"max": 0.40, "min": 0.2},
        "LWR": {"max": 0.25, "min": 0.05},
        'FRE': {"max": 50.0, "min": -15.0},
        "TTR": {"max": 0.8, "min": 0.4},
        "DRV": {"max": 35.0, "min": 20.0},
        "PS": {"max": 0.16, "min": 0.08},
        "SS": {"max": 0.52, "min": 0.40},
        "SCD": {"max": 1.6, "min": 0.0},
        
    }
    
    for i in ['Abstract', 'Meta-Review', 'Review']:
        n_rows, n_cols = 1, 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3))
        for idx, feature in enumerate(features.keys()):
            print(idx)
            row, col = divmod(idx, n_cols)
            ax = axes[col]

            if feature in ['AWL', 'ASL', 'SWR', 'LWR', 'TTR']:
                df_filtered = df[df[feature] != 0.0]
                df_filtered = df
            else:
                df_filtered = df
            
            df_filtered = df_filtered[df_filtered['writer'] == 'Human']
            df_filtered = df_filtered[df_filtered['type'] == i]
            # plt.figure(figsize=(5.5, 5.5))
            

            desired_order=['≤2019', '2020', '2021', '2022', '2023', '2024']
            df_filtered["year"] = pd.Categorical(df_filtered["year"], categories=desired_order, ordered=True)

            sns.lineplot(
                data=df_filtered,
                x="year", y=feature, palette=cm.colors, ax=ax,)
            

            
            # ax.set_ylim(bottom=features_value[feature]['min'], top=features_value[feature]['max'])
            # ax.set_title(features[feature], fontsize=17)
            # ax.set_ylabel(feature,fontsize=17)
            ax.set_title(features[feature])
            ax.set_ylabel(feature)
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelsize=10)
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
        plt.savefig(f'LLM_penetration/figure/output/feature_trend_{i}.png', dpi=300, bbox_inches="tight")

        print('a')





if __name__ == "__main__":
    main()
