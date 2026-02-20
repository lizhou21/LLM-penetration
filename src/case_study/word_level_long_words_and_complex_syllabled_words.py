import json
from nltk.tokenize import SyllableTokenizer


def count_syllables(word):
    syllable_tokenizer = SyllableTokenizer()
    return len(syllable_tokenizer.tokenize(word))


def is_complex_word(word):
    return count_syllables(word) >= 3


def is_long_word(word):
    return len(word) >= 10


def process_words_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    complex_word_count = 0
    long_word_count = 0
    word_count = len(data)

    for item in data:
        word = item.get('word')
        if word:
            # 判断是否为复杂词
            if is_complex_word(word):
                complex_word_count += 1

            # 判断是否为长单词
            if is_long_word(word):
                long_word_count += 1

    # 计算各项指标
    if word_count > 0:
        complex_word_ratio = complex_word_count / word_count
        long_word_ratio = long_word_count / word_count

        print(file_path)
        print(f"complex-syllabled words: {complex_word_ratio * 100:.2f}%")
        print(f"long words: {long_word_ratio * 100:.2f}%")


def main():
    file_path = ''
    process_words_from_json(file_path)


if __name__ == "__main__":
    main()
