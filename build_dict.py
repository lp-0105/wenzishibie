import pandas as pd
import os

# 1. 提取所有可能的文字
# 针对你的 CSV 大概率是 GBK/GB18030 编码进行处理
try:
    df = pd.read_csv('data/train_label.csv', encoding='gb18030')
except:
    df = pd.read_csv('data/train_label.csv', encoding='utf-8')

all_text = "".join(df['value'].astype(str).tolist())
unique_chars = sorted(list(set(all_text)))

# 2. 写入 ppocr_keys_v1.txt
with open('ppocr_keys_v1.txt', 'w', encoding='utf-8') as f:
    for char in unique_chars:
        f.write(char + '\n')

print(f"提取出 {len(unique_chars)} 个唯一字符，已保存到 ppocr_keys_v1.txt")
