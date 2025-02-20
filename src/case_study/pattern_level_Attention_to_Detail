import json
import re
from tqdm import tqdm


def clean_text(text):
    cleaned_text = re.sub(r'[\r\n*\\]+', ' ', text)
    return cleaned_text


def count_patterns(json_file_path):
    total_metareviews = 0
    total_https_count = 0
    metareviews_with_https = 0

    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in tqdm(data, desc="Processing comments", unit="comment"):
            total_metareviews += 1
            try:
                comment = item["meta_reivew"]['comment']
            except KeyError:
                print(f"Warning: Item {total_metareviews} does not have 'meta_review_claude' key. Skipping...")
                continue
            cleaned_comment = clean_text(comment)
            # 使用正则表达式查找 https
            https_matches = re.findall(r'http', cleaned_comment)
            https_count = len(https_matches)
            total_https_count += https_count
            if https_count > 0:
                metareviews_with_https += 1

    print(f"FR_Attention_to_Detail: {metareviews_with_https / total_metareviews * 100:.2f}")
    print(f"FI_Attention_to_Detail: {total_https_count / metareviews_with_https:.2f}")


def main():
    json_file_path = 'ICLR_2017_2019_pol_seg.json'
    count_patterns(json_file_path)


if __name__ == "__main__":
    main()
