import os
import json
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import math
from scipy.stats import t
import spacy

# 加载英文模型
nlp = spacy.load('en_core_web_sm')

import string
stop_words = set(stopwords.words('english'))

text = "The quick brown fox jumps over the lazy dog"
doc = nlp(text)

# 输出每个单词的词性
for token in doc:
    print(f"{token.text}: {token.pos_}")

data_type = 'Abs'
alpha=0.05
alpha_smooth=1

data_dir = '/mntcephfs/data/haizhouli/Lab-projects/lizhou/LLM_penetration/data'
save_dir = '/mntcephfs/data/haizhouli/Lab-projects/lizhou/LLM_penetration/results/prefer_word'

dataset_file = os.path.join(data_dir, 'ICLR_2017_2019_pol_seg.json')
with open(dataset_file, "r", encoding="utf-8") as f:
    
    data = json.load(f)
    

All_word = {
    'human': [],
    'gpt4': [],
    'claude': [],
    'gemini': []
}

All_word_per_text = {
    'human': [],
    'gpt4': [],
    'claude': [],
    'gemini': []
}

All_word_set = {
    'human': [],
    'gpt4': [],
    'claude': [],
    'gemini': []
}

compair_word_set = {
    'gpt4': [],
    'claude': [],
    'gemini': []
}

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
        return tag


for d in tqdm(data):
    if data_type == 'Abs':
        human_written = d['abstract']
        gpt4_written = d['polish_abstract']
        claude_written = d['polish_abstract_claude']
        gemini_written = d['polish_abstract_claude']
    else:
        human_written = d['meta_reivew']['comment']
        gpt4_written = d['meta_review_gpt']
        claude_written = d['meta_review_claude']
        gemini_written = d['meta_review_claude']
        ss = 'Our results demonstrate that both label smoothing and our confidence penalty enhance the performance of state-of-the-art models on these benchmarks without altering existing hyper-parameters.'
        a=pos_tag(word_tokenize(ss))
        doc1 = [(token.text, token.pos_) for token in doc]
     
        
    human_words = [(token.text.lower(), token.pos_) for token in nlp(human_written) if token.text.lower() not in stop_words and token.text not in string.punctuation]
    human_words_map = [(word[0], map_pos(word[1])) for word in human_words]
    All_word_per_text['human'].append([word for word in human_words_map])

    gpt4_words = [(token.text.lower(), token.pos_) for token in nlp(gpt4_written) if token.text.lower() not in stop_words and token.text not in string.punctuation]
    gpt4_words_map = [(word[0], map_pos(word[1])) for word in gpt4_words]
    All_word_per_text['gpt4'].append([word for word in gpt4_words_map])

    claude_words = [(token.text.lower(), token.pos_) for token in nlp(claude_written) if token.text.lower() not in stop_words and token.text not in string.punctuation]
    claude_words_map = [(word[0], map_pos(word[1])) for word in claude_words]
    All_word_per_text['claude'].append([word for word in claude_words_map])

    gemini_words = [(token.text.lower(), token.pos_) for token in nlp(gemini_written) if token.text.lower() not in stop_words and token.text not in string.punctuation]
    gemini_words_map = [(word[0], map_pos(word[1])) for word in gemini_words]
    All_word_per_text['gemini'].append([word for word in gemini_words_map])
    
    # human_words = [word.lower() for word in word_tokenize(human_written) if word.lower() not in stop_words and word not in string.punctuation]
    # gpt4_words = [word.lower() for word in word_tokenize(gpt4_written) if word.lower() not in stop_words and word not in string.punctuation]
    # claude_words = [word.lower() for word in word_tokenize(claude_written) if word.lower() not in stop_words and word not in string.punctuation]
    # gemini_words = [word.lower() for word in word_tokenize(gemini_written) if word.lower() not in stop_words and word not in string.punctuation]
    All_word['human'].extend(human_words_map)
    All_word['gpt4'].extend(gpt4_words_map)
    All_word['claude'].extend(claude_words_map)
    All_word['gemini'].extend(gemini_words_map)
    
All_word_set['human'] = set(All_word['human'])
All_word_set['gpt4'] = set(All_word['gpt4'])
All_word_set['claude'] = set(All_word['claude'])
All_word_set['gemini'] = set(All_word['gemini'])

compair_word_set['gpt4'] = list(All_word_set['gpt4'].union(All_word_set['human']))
compair_word_set['claude'] = list(All_word_set['claude'].union(All_word_set['human']))
compair_word_set['gemini'] = list(All_word_set['gemini'].union(All_word_set['human']))


test_word = {
    'gpt4': [],
    'claude': [],
    'gemini': []
}

for pair, vocab in compair_word_set.items():
    for v in tqdm(vocab):
        all_cnt_human = len(All_word_per_text['human'])
        all_cnt_llm = len(All_word_per_text[pair])

        cnt_human = sum([1 if v in per else 0 for per in All_word_per_text['human']]) + alpha_smooth
        cnt_llm = sum([1 if v in per else 0 for per in All_word_per_text[pair]]) + alpha_smooth

        # cnt_human = All_word['human'].count(v) + alpha_smooth
        # cnt_llm = All_word[pair].count(v) + alpha_smooth
        p_hat_human = cnt_human / all_cnt_human
        p_hat_llm = cnt_llm / all_cnt_llm
        
        p_hat = (cnt_human + cnt_llm) / (all_cnt_human + all_cnt_llm)

        se_human = math.sqrt(p_hat_human * (1 - p_hat_human) / all_cnt_human)
        se_llm = math.sqrt(p_hat_llm * (1 - p_hat_llm) / all_cnt_llm)
        se = math.sqrt(se_human ** 2 + se_llm ** 2)
        # 计算 t 统计量
        t_stat = (p_hat_llm - p_hat_human) / se
        # 计算自由度，使用 Welch's t-test 近似

        se_2_n_human =  (se_human ** 2) / all_cnt_human
        se_2_n_llm =  (se_llm ** 2) / all_cnt_llm
        df_child = (se_2_n_human + se_2_n_llm) ** 2
        df_parent = (se_2_n_human ** 2)/(all_cnt_human-1) + (se_2_n_llm ** 2)/(all_cnt_llm-1)

        df =  df_child / df_parent
        # 计算 t 分布的临界值（右侧单侧检验）
        t_critical = t.ppf(1 - alpha, df)
        # 判断是否拒绝原假设（只考虑 AI 显著高于人类）
        reject_null = t_stat > t_critical
        if reject_null:
            test_word[pair].append({
                "word": v[0],
                "pos": v[1],
                "t_score": t_stat,
                "t_critical": t_critical,
                "human_count": cnt_human - alpha_smooth,
                "llm_count": cnt_llm - alpha_smooth,
                "human_p": p_hat_human,
                "llm_p": p_hat_llm
            })

    test_word[pair].sort(key=lambda x: x["t_score"], reverse=True)
    save_file = os.path.join(save_dir, f'spacy_{data_type}_{pair}_word_new.json')
    with open(save_file, "w") as f:
        json.dump(test_word[pair], f, indent=4)


print('a')