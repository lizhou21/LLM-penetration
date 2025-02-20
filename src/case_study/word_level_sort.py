
import os
import json

def process_json_file(filename, pos_list, top_n):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            # 加载 JSON 数据
            data = json.load(file)
            # 初始化每个词性的列表
            pos_data = {pos: [] for pos in pos_list}

            # 遍历每个条目
            for entry in data:
                pos = entry.get('pos')
                if pos in pos_list:
                    # 计算新属性的值
                    try:
                        ratio = (entry['llm_count'] - entry['human_count']) / (entry['human_count'] + 1)
                    except ZeroDivisionError:
                        ratio = 0
                    # 添加新属性
                    entry['ratio'] = ratio
                    # 将条目添加到对应的词性列表中
                    pos_data[pos].append(entry)

            # 初始化一个空列表用于存储每个文件处理后的最终结果
            final_data = []
            # 对每个词性的列表按新属性从高到低排序，并取前 top_n 个
            for pos in pos_list:
                pos_data[pos].sort(key=lambda x: x['ratio'], reverse=True)
                final_data.extend(pos_data[pos][:top_n])

            # 生成新的文件名
            base_name, ext = os.path.splitext(filename)
            new_filename = f"{base_name}_processed{ext}"

            # 将最终结果保存到一个新的 JSON 文件中
            with open(new_filename, 'w', encoding='utf-8') as new_file:
                json.dump(final_data, new_file, ensure_ascii=False, indent=4)
    except json.JSONDecodeError:
        print(f"Error decoding JSON file: {filename}")


def main():
    # 定义词性列表
    pos_list = ['ADJ', 'ADV', 'VERB', 'NOUN']
    # 定义每个词性保留的数量
    top_n = 30

    # 遍历当前目录下的所有文件
    for filename in os.listdir('.'):
        if filename.endswith('.json'):
            process_json_file(filename, pos_list, top_n)


if __name__ == "__main__":
    main()
