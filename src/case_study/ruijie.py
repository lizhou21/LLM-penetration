import json
import re
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from tqdm import tqdm
import math
from scipy.stats import t
import os

def clean_text(text):
    #cleaned_text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    cleaned_text = re.sub(r'[\r\n*\\]+', ' ', text)
    return cleaned_text


def map_pos(tag):
    # 将细粒度的词性标签映射为粗粒度的标签，只保留名词、动词、形容词、副词、代词
    pos_mapping = {
        'NN': 'NOUN',
        'NNS': 'NOUN',
        'NNP': 'NOUN',
        'NNPS': 'NOUN',
        'VB': 'VERB',
        'VBD': 'VERB',
        'VBG': 'VERB',
        'VBN': 'VERB',
        'VBP': 'VERB',
        'VBZ': 'VERB',
        'JJ': 'ADJ',
        'JJR': 'ADJ',
        'JJS': 'ADJ',
        'RB': 'ADV',
        'RBR': 'ADV',
        'RBS': 'ADV',
        'PRP': 'PRONOUN',
        'PRP$': 'PRONOUN',
        'WP': 'PRONOUN',
        'WP$': 'PRONOUN'
    }
    if tag in pos_mapping:
        return pos_mapping[tag]
    else:
        return None


def t_test(data, key_human, key_ai, alpha=0.05, alpha_smooth=1):
    word_counts_human = {'ADJ': {}, 'VERB': {}, 'ADV': {}, 'NOUN': {}, 'PRONOUN': {}}
    word_counts_ai = {'ADJ': {}, 'VERB': {}, 'ADV': {}, 'NOUN': {}, 'PRONOUN': {}}
    # 遍历数据，统计人类和 AI 文本中每个词的出现次数并进行词性标注
    for item in tqdm(data, desc="Counting words"):
        if key_human in item:
            text = item[key_human]['comment']
            cleaned_text = clean_text(text)
            tagged_words_human = pos_tag(word_tokenize(cleaned_text))
            for word, tag in tagged_words_human:
                mapped_tag = map_pos(tag)
                if mapped_tag in word_counts_human:
                    # 将单词转换为小写
                    word = word.lower()
                    if word in word_counts_human[mapped_tag]:
                        word_counts_human[mapped_tag][word] += 1
                    else:
                        word_counts_human[mapped_tag][word] = 1
        if key_ai in item:
            text = item[key_ai]
            cleaned_text = clean_text(text)
            tagged_words_ai = pos_tag(word_tokenize(cleaned_text))
            for word, tag in tagged_words_ai:
                mapped_tag = map_pos(tag)
                if mapped_tag in word_counts_ai:
                    # 将单词转换为小写
                    word = word.lower()
                    if word in word_counts_ai[mapped_tag]:
                        word_counts_ai[mapped_tag][word] += 1
                    else:
                        word_counts_ai[mapped_tag][word] = 1
    results = {'ADJ': [], 'VERB': [], 'ADV': [], 'NOUN': [], 'PRONOUN': []}
    # 假设检验
    for pos in word_counts_human.keys():
        all_words = set(word_counts_human[pos].keys()) | set(word_counts_ai[pos].keys())
        n_human = sum(word_counts_human[pos].values()) + alpha_smooth * len(all_words)
        n_ai = sum(word_counts_ai[pos].values()) + alpha_smooth * len(all_words)
        for word in tqdm(all_words, desc=f"Hypothesis testing ({pos})"):
            x_human = word_counts_human[pos].get(word, 0) + alpha_smooth
            x_ai = word_counts_ai[pos].get(word, 0) + alpha_smooth
            p_hat_human = x_human / n_human
            p_hat_ai = x_ai / n_ai
            p_hat = (x_human + x_ai) / (n_human + n_ai)
            se_human = math.sqrt(p_hat_human * (1 - p_hat_human) / n_human)
            se_ai = math.sqrt(p_hat_ai * (1 - p_hat_ai) / n_ai)
            se = math.sqrt(se_human ** 2 + se_ai ** 2)
            # 计算 t 统计量
            t_stat = (p_hat_ai - p_hat_human) / se
            # 计算自由度，使用 Welch's t-test 近似
            df = ((se_human ** 2 / n_human + se_ai ** 2 / n_ai) ** 2) / ((se_human ** 2 / n_human) ** 2 / (n_human - 1) + (se_ai ** 2 / n_ai) ** 2 / (n_ai - 1))
            # 计算 t 分布的临界值（右侧单侧检验）
            t_critical = t.ppf(1 - alpha, df)
            # 判断是否拒绝原假设（只考虑 AI 显著高于人类）
            reject_null = t_stat > t_critical
            if reject_null:
                results[pos].append({
                    "word": word,
                    "t_score": t_stat,
                    "t_critical": t_critical,
                    "human_count": x_human - alpha_smooth,
                    "ai_count": x_ai - alpha_smooth
                })
        # 按照 t_score 从高到低排序
        results[pos].sort(key=lambda x: x["t_score"], reverse=True)
    return results


def main():
    # 读取 JSON 文件
    data_dir = '/mntcephfs/data/haizhouli/Lab-projects/lizhou/LLM_penetration/data'

    dataset_file = os.path.join(data_dir, 'ICLR_2017_2019_pol_seg.json')
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    key_human = 'meta_reivew'
    key_ai = 'meta_review_claude'
    significant_results = t_test(data, key_human, key_ai, alpha=0.0001)
    # 将结果保存为 JSON 文件
    for pos, pos_results in significant_results.items():
        with open(f'results_{pos}.json', 'w') as outfile:
            json.dump(pos_results, outfile, indent=4)


if __name__ == "__main__":
    main()