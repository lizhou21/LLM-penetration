
import json
import argparse
from tqdm import tqdm
import spacy
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")

All_types = ['abstract', 'meta', 'review']
# 读取 Excel 文件



def main():
    df = pd.read_excel('LLM_penetration/results/detection/review_trend.xlsx')  # sheet_name 可以指定要读取的表单



    cm = plt.get_cmap('tab10')
    # features = {
    #     "ScholarDetect-abs": "ScholarDetect-abs", 
    #     "ScholarDetect-meta": "ScholarDetect-meta",
    #     "ScholarDetect-mix": "ScholarDetect-mix",        
    # }

    n_rows, n_cols = 1, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 4))

    offset_dict = {
    "Abs": -0.08,
    "Meta": 0.0,
    "Hybrid": 0.08,
}


    for idx, data_type in enumerate(['Meta-Review', 'Review']):
        row, col = divmod(idx, n_cols)
        ax = axes[col]
        df_filtered = df[df['Data Type'] == data_type]
        df_filtered["Year_offset"] = df_filtered["Year"] + df_filtered["Model"].map(offset_dict)

        sns.lineplot(
                data=df_filtered,
                x="Year_offset", y='LLM penetration (%)', hue='Model', 
                style="Model",
                markers = ['o', 's', 'D'],  # 指定点标记：圆形、方形、菱形、三角形、五角星等
                markeredgecolor="black",
                # dashes=[(2, 2), '', (2, 2), '', (2, 2), ''],  # 指定线型：虚线和实线的混合
                palette=cm.colors, alpha=0.80, ax=ax)
        
        ax.set_ylim(top=17.0)
        ax.set_title(f"Pediction on {data_type}")
        ax.set_ylabel("LLM Penetration (%)")
        ax.set_xlabel("Year")
        ax.set_xticks(np.unique(df["Year"]))




    plt.tight_layout()
    plt.savefig(f'LLM_penetration/figure/output/detection_trend_review.png', dpi=300, bbox_inches="tight")

    print('a')





if __name__ == "__main__":
    main()
