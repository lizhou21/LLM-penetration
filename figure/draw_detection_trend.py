
import json
import argparse
from tqdm import tqdm
import spacy
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

All_types = ['abstract', 'meta', 'review']
# 读取 Excel 文件



def main():
    df = pd.read_excel('LLM_penetration/results/detection/binary_trend.xlsx')  # sheet_name 可以指定要读取的表单



    cm = plt.get_cmap('Set2')
    features = {
        "ScholarDetect-abs": "ScholarDetect-abs", 
        "ScholarDetect-meta": "ScholarDetect-meta",
        "ScholarDetect-mix": "ScholarDetect-mix",        
    }
    6
    for i in ['Abstract', 'Meta-Review', 'Review']:
        n_rows, n_cols = 1, 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3))
        for idx, feature in enumerate(features.keys()):
            print(idx)
            row, col = divmod(idx, n_cols)
            ax = axes[col]

            df_filtered = df
            

            df_filtered = df_filtered[df_filtered['Type'] == i]
            df_filtered = df_filtered[df_filtered['Model'] == feature]
            # plt.figure(figsize=(5.5, 5.5))
            

            # desired_order=['2020', '2021', '2022', '2023', '2024']
            # df_filtered["Year"] = pd.Categorical(df_filtered["Year"], categories=desired_order, ordered=True)

            sns.lineplot(
                data=df_filtered,
                x="Year", y='Percentage', palette=cm.colors, ax=ax,)
            

            
            # ax.set_ylim(bottom=features_value[feature]['min'], top=features_value[feature]['max'])
            ax.set_title(features[feature])
            ax.set_ylabel("Percentage (%)")
            ax.set_xlabel('')
            ax.tick_params(axis='x')
            ax.tick_params(axis='y')

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
        plt.savefig(f'LLM_penetration/figure/output/detection_trend_{i}.png', dpi=300, bbox_inches="tight")

        print('a')





if __name__ == "__main__":
    main()
