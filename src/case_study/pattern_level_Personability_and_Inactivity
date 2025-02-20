import json
import nltk
import re
from tqdm import tqdm


def clean_text(text):
    cleaned_text = re.sub(r'[\r\n*\\]+', ' ', text)
    return cleaned_text


def count_reviewer_mentions(json_file_path):
    total_metareviews = 0
    count_first_person_metareviews = 0
    count_question_metareviews = 0
    total_first_person_sentences = 0
    total_question_sentences = 0

    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in tqdm(data, desc="Processing comments", unit="comment"):
            total_metareviews += 1
            comment = item["meta_review_claude"]
            # 清理文本
            cleaned_comment = clean_text(comment)
            # 使用nltk将评论分割成句子
            sentences = nltk.sent_tokenize(cleaned_comment)

            has_first_person = False
            has_question = False
            first_person_count = 0
            question_count = 0

            for sentence in sentences:
                # 判断是否为第一人称句子
                tokens = nltk.word_tokenize(sentence)
                first_person_found = any(token.lower() in ['i', 'we'] for token in tokens)
                if first_person_found:
                    first_person_count += 1
                    has_first_person = True
                # 判断是否为问句
                if sentence.strip().endswith('?'):
                    question_count += 1
                    has_question = True

            if has_first_person:
                count_first_person_metareviews += 1
                total_first_person_sentences += first_person_count
            if has_question:
                count_question_metareviews += 1
                total_question_sentences += question_count


    print(f"FR_Personability: {count_first_person_metareviews / total_metareviews * 100:.2f}")
    print(f"FR_Inactivity: {count_question_metareviews / total_metareviews * 100:.2f}")

    if count_first_person_metareviews > 0:
        avg_first_person = total_first_person_sentences / count_first_person_metareviews
        print(f"FI_Personability： {avg_first_person:.2f}")
    else:
        print(f"FI_Personability： 0.00")

    if count_question_metareviews > 0:
        avg_question = total_question_sentences / count_question_metareviews
        print(f"FI_Inactivity： {avg_question:.2f}")
    else:
        print(f"FI_Inactivity： 0.00")


def main():
    # 请将此处的文件路径替换为你实际的ICLR_2017_2019_pol_seg.json文件路径
    json_file_path = 'ICLR_2017_2019_pol_seg.json'
    count_reviewer_mentions(json_file_path)


if __name__ == "__main__":
    main()
