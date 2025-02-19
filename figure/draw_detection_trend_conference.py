
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
    df = pd.read_excel('LLM_penetration/results/detection/conference_trend.xlsx')  # sheet_name 可以指定要读取的表单



    cm = plt.get_cmap('tab10')
    # features = {
    #     "ScholarDetect-abs": "ScholarDetect-abs", 
    #     "ScholarDetect-meta": "ScholarDetect-meta",
    #     "ScholarDetect-mix": "ScholarDetect-mix",        
    # }
    # plt.figure(figsize=(6, 4.5))  # 宽 10 英寸，高 6 英寸
    plt.figure(figsize=(7, 3.5))  # 宽 10 英寸，高 6 英寸

    offset_dict = {
    "ICLR": -0.09,
    "ACL": -0.06,
    "CVPR": -0.03,
    "EMNLP": 0.0,
    "ICML": 0.03,
    "IJCAI": 0.06,
    "NeurIPS": 0.09,
}

# 根据偏移量调整横坐标
    df["Year_offset"] = df["Year"] + df["Conference"].map(offset_dict)

    
    sns.lineplot(
            data=df,
            x="Year_offset", y='LLM penetration (%)', hue='Conference', 
            style="Conference",
            markers = ['o', 's', 'D', '^', 's', 'D', '^'],  # 指定点标记：圆形、方形、菱形、三角形、五角星等
            markeredgecolor="black",
            dashes=[(2, 2), '', (2, 2), '', (2, 2), ''],  # 指定线型：虚线和实线的混合
            palette=cm.colors, alpha=0.80)
    

    plt.title(r"$\text{ScholarDetect}_{\text{Hybrid}}$ Prediction on Abstract")
    plt.xlabel("Year")
    plt.ylabel("LLM Penetration (%)")
    plt.legend(title="Conference")
    plt.xticks(np.unique(df["Year"]))
    plt.show()
    plt.tight_layout()
    plt.savefig(f'LLM_penetration/figure/output/detection_trend_conference.png', dpi=300, bbox_inches="tight")

    print('a')





if __name__ == "__main__":
    main()
