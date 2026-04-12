import os
import pandas as pd
import random
import zipfile
from shutil import copy2, move

def unzip_data(zip_path, target_dir):
    """
    解压数据集文件
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path} to {target_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("Extraction completed.")
    else:
        print(f"Warning: Zip file {zip_path} not found.")

def prepare_paddleocr_data(csv_path, img_dir, output_dir, train_ratio=0.9):
    # 1. 创建目标目录
    train_data_dir = os.path.join(output_dir, "train_data")
    rec_img_dir = os.path.join(train_data_dir, "train_images")
    os.makedirs(rec_img_dir, exist_ok=True)

    # 2. 读取原始 CSV 标签
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, encoding='gbk')
    
    # 假设列名为 name (文件名) 和 value (文本内容)
    # 根据用户提供的预览：name,value
    
    results = []
    all_chars = set()

    print("Checking images and collecting labels...")
    for index, row in df.iterrows():
        img_name = str(row['name'])
        label = str(row['value'])
        
        src_img_path = os.path.join(img_dir, img_name)
        
        # 3. 健壮性检查：图片是否存在
        if os.path.exists(src_img_path):
            # 将图片复制到目标目录，或者使用软链接 (os.symlink)
            # 这里为了简单直接复制，以后建议改为软链接
            dst_img_path = os.path.join(rec_img_dir, img_name)
            if not os.path.exists(dst_img_path):
                copy2(src_img_path, dst_img_path)
            
            # PaddleOCR 格式: 相对路径 + \t + 文本
            # 路径相对于后续 PaddleOCR 配置中的 root 目录，这里统一放在 train_images 下
            line = f"train_images/{img_name}\t{label}"
            results.append(line)
            
            # 统计字符用于生成字典
            for char in label:
                all_chars.add(char)
        else:
            print(f"Warning: Image {src_img_path} not found, skipping.")

    # 4. 数据集随机打乱并划分 (使用固定种子确保可重复运行结果一模一样)
    random.seed(42)
    random.shuffle(results)
    split_idx = int(len(results) * train_ratio)
    train_list = results[:split_idx]
    val_list = results[split_idx:]

    # 5. 写入映射文件
    with open(os.path.join(train_data_dir, "train.txt"), "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line + "\n")
            
    with open(os.path.join(train_data_dir, "val.txt"), "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line + "\n")

    # 6. 生成字典文件 dict.txt
    dict_list = sorted(list(all_chars))
    with open(os.path.join(train_data_dir, "dict.txt"), "w", encoding="utf-8") as f:
        for char in dict_list:
            f.write(char + "\n")

    print(f"Finished! Total images: {len(results)}")
    print(f"Train samples: {len(train_list)}")
    print(f"Val samples: {len(val_list)}")
    print(f"Dictionary size: {len(dict_list)}")
    print(f"Output directory: {train_data_dir}")

if __name__ == "__main__":
    # 本地环境路径配置
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 定义解压后存放数据的目录 (由 extract_data.py 生成)
    # 假设解压后 data/ 目录就在当前目录下
    DATA_ROOT = os.path.join(CURRENT_DIR, "data")
    IMAGE_DIR = os.path.join(DATA_ROOT, "train_images")
    CSV_FILE = os.path.join(DATA_ROOT, "train_label.csv")
    
    # 2. 检查数据是否存在
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(CSV_FILE):
        print(f"错误: 找不到数据目录或标签文件。")
        print(f"请先运行 'python extract_data.py' 来恢复 data/ 文件夹。")
    else:
        # 3. 设置输出目录（训练数据将生成在当前目录下的 train_data 文件夹内）
        OUTPUT_ROOT = CURRENT_DIR
        
        # 4. 运行数据准备逻辑
        print("开始准备训练数据...")
        prepare_paddleocr_data(CSV_FILE, IMAGE_DIR, OUTPUT_ROOT)
        print("数据准备完成！")
