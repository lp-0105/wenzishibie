import os
import pandas as pd
import numpy as np
import paddle
from PIL import Image
from model import TransformerOCR

# --- 1. 配置 ---
IMG_H, IMG_W = 32, 320
TEST_DIR = 'data/test_images'
SAVE_FILE = 'result.txt'  # 修改为 .txt 后缀
# 自动寻找最新的权重文件
CHECKPOINT_DIR = 'checkpoints'

# --- 2. 加载字典 ---
if not os.path.exists('ppocr_keys_v1.txt'):
    print("错误: 找不到 ppocr_keys_v1.txt")
    exit(1)

with open('ppocr_keys_v1.txt', 'r', encoding='utf-8') as f:
    chars = [line.strip('\n') for line in f.readlines() if line.strip('\n')]
id_to_char = {i + 1: char for i, char in enumerate(chars)}
num_classes = len(chars) + 1

# --- 3. 加载模型 ---
def get_latest_checkpoint(path):
    # 优先使用验证集表现最好的模型
    best_path = os.path.join(path, 'best_model.pdparams')
    if os.path.exists(best_path):
        return best_path
        
    files = [f for f in os.listdir(path) if f.endswith('.pdparams')]
    if not files: return None
    # 按照 epoch 数字排序
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return os.path.join(path, files[-1])

latest_model = get_latest_checkpoint(CHECKPOINT_DIR)
if not latest_model:
    print(f"错误: 在 {CHECKPOINT_DIR} 下没找到模型权重")
    exit(1)

print(f"正在加载最优权重: {latest_model}")
model = TransformerOCR(num_classes)
model.set_state_dict(paddle.load(latest_model))
model.eval()

# --- 4. 预测函数 (带 CTC Greedy 解码) ---
def decode_ctc(preds_idx):
    res = []
    last_idx = 0
    for idx in preds_idx:
        if idx > 0 and idx != last_idx: # 0 是 blank, 且去重
            res.append(id_to_char.get(idx, ''))
        last_idx = idx
    return "".join(res)

def predict_batch(image_paths):
    imgs = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('L')
            img = img.resize((IMG_W, IMG_H))
            img = np.array(img).astype('float32') / 255.0
            imgs.append(img[np.newaxis, :, :])
        except:
            imgs.append(np.zeros((1, IMG_H, IMG_W), dtype='float32'))
    
    imgs_tensor = paddle.to_tensor(np.array(imgs))
    with paddle.no_grad():
        # 兼容最新模型：返回 ctc_out 和 att_out，只需取 ctc_out
        ctc_out, _ = model(imgs_tensor) 
    
    preds_idx = paddle.argmax(ctc_out, axis=-1).numpy()
    return [decode_ctc(p) for p in preds_idx]

# --- 5. 循环预测测试集 ---
print("开始对测试集图片进行预测...")
# 按数字顺序排序文件名，例如 0.jpg, 1.jpg, 2.jpg...
def sort_key(filename):
    name = filename.split('.')[0]
    if name.isdigit():
        return int(name)
    return name

test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
test_files.sort(key=sort_key) 

all_results = []
batch_size = 64
for i in range(0, len(test_files), batch_size):
    batch_files = test_files[i : i + batch_size]
    batch_paths = [os.path.join(TEST_DIR, f) for f in batch_files]
    
    batch_preds = predict_batch(batch_paths)
    
    for f, p in zip(batch_files, batch_preds):
        all_results.append({'new_name': f, 'value': p}) # 修改列名为 'new_name'
    
    if (i // batch_size) % 10 == 0:
        print(f"进度: {i}/{len(test_files)}")

# --- 6. 按照官方要求格式保存 (制表符分割的 .txt) ---
df = pd.DataFrame(all_results)
# 1. 确保表头为 new_name 和 value
# 2. 使用 \t (制表符) 分割，避免内容中的逗号干扰
# 3. 后缀为 .txt，编码为 utf-8
df.to_csv(SAVE_FILE, sep='\t', index=False, encoding='utf-8')
print(f"所有预测完成，符合提交要求的格式文件已保存至: {SAVE_FILE}")
