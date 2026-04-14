import os
import pandas as pd
import numpy as np
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.io import Dataset, DataLoader
from PIL import Image, ImageFilter
import random
import time
import subprocess
import warnings
from model import TransformerOCR

warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 参数配置 (无人房间·稳健冲刺版: 200轮速成架构) ---
IMG_H, IMG_W = 32, 320 
# 调整物理 Batch Size 到 100，确保 8GB 显存有充足的“呼吸空间”，不卡死
BATCH_SIZE = 100  
# 梯度累加步数：累计 2 次更新一次参数 (有效 Batch Size = 200)
ACCUM_STEPS = 2
EPOCHS = 200      
DEVICE = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'

def get_gpu_info():
    try:
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,nounits,noheader"])
        res = res.decode('utf-8').strip().split(',')
        return f"GPU Mem: {res[0].strip()}/{res[1].strip()}MB, Util: {res[2].strip()}%"
    except:
        return "GPU Info N/A"

class OCRTransforms:
    def __init__(self, mode='none'):
        self.mode = mode
    def __call__(self, img):
        if self.mode == 'train':
            # 1. 基础旋转 (稍微加大到 -7 to 7 度)
            if random.random() > 0.5:
                angle = random.uniform(-7, 7)
                img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
            
            # 2. 空间扰动 (轻微透视/扭曲，强制 TPS 学习纠偏)
            if random.random() > 0.7:
                w, h = img.size
                x1, y1 = random.uniform(0, 0.05) * w, random.uniform(0, 0.05) * h
                x2, y2 = w - random.uniform(0, 0.05) * w, random.uniform(0, 0.05) * h
                x3, y3 = w - random.uniform(0, 0.05) * w, h - random.uniform(0, 0.05) * h
                x4, y4 = random.uniform(0, 0.05) * w, h - random.uniform(0, 0.05) * h
                img = img.transform((w, h), Image.QUAD, (x1, y1, x4, y4, x3, y3, x2, y2), fillcolor=255)

            # 3. 随机噪点 (模拟脏背景/漏墨，浓度设低 0.02)
            if random.random() > 0.8:
                img_array = np.array(img)
                mask = np.random.uniform(0, 1, img_array.shape) < 0.01 # 1% 的像素变黑/点
                img_array[mask] = 0
                img = Image.fromarray(img_array)

            # 4. 模糊与对比度 (保持原有逻辑)
            if random.random() > 0.8:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
            from PIL import ImageEnhance
            if random.random() > 0.7:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))
        return img

with open('ppocr_keys_v1.txt', 'r', encoding='utf-8') as f:
    chars = [line.strip('\n') for line in f.readlines() if line.strip('\n')]
char_to_id = {char: i + 1 for i, char in enumerate(chars)}
num_classes = len(chars) + 1

class CustomOCRDataset(Dataset):
    def __init__(self, txt_path, img_dir, mode='none'):
        self.data = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.data.append(parts)
        self.img_dir = img_dir
        self.mode = mode
        self.trans = OCRTransforms(mode)
    def __getitem__(self, idx):
        img_rel_path, label_text = self.data[idx]
        img_path = os.path.join(self.img_dir, img_rel_path)
        try:
            img = Image.open(img_path).convert('L')
            if self.mode != 'none': img = self.trans(img)
            img = img.resize((IMG_W, IMG_H))
            img = np.array(img).astype('float32') / 255.0
            img = img[np.newaxis, :, :] 
            label = [char_to_id.get(c, 0) for c in label_text]
            return img, np.array(label, dtype='int32'), len(label)
        except:
            return self.__getitem__(np.random.randint(0, len(self.data)))
    def __len__(self): return len(self.data)

def collate_fn(batch):
    imgs, labels, lens = zip(*batch)
    max_len = max(lens)
    padded_labels = [np.pad(l, (0, max_len - len(l)), 'constant') for l in labels]
    return np.array(imgs), np.array(padded_labels), np.array(lens)

def decode_ctc(preds_idx):
    res = []
    last_idx = 0
    for idx in preds_idx:
        if idx > 0 and idx != last_idx: res.append(idx)
        last_idx = idx
    return res

def train():
    # 自动获取当前脚本所在目录，确保在 Colab/本地 运行路径一致
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DATA_ROOT = os.path.join(BASE_DIR, 'train_data')
    TRAIN_TXT = os.path.join(TRAIN_DATA_ROOT, 'train.txt')
    VAL_TXT = os.path.join(TRAIN_DATA_ROOT, 'val.txt')

    paddle.set_device(DEVICE)
    
    # 检查训练数据是否存在
    if not os.path.exists(TRAIN_TXT):
        print(f"错误: 找不到训练数据 {TRAIN_TXT}")
        print("请确保已运行 python extract_data.py 和 python prepare_data.py")
        return

    train_ds = CustomOCRDataset(TRAIN_TXT, TRAIN_DATA_ROOT, mode='train')
    val_ds = CustomOCRDataset(VAL_TXT, TRAIN_DATA_ROOT, mode='none')
    
    # 无人房间模式：把多进程读图拉满到 8，并且开启共享内存加速
    num_workers = 8 if DEVICE == 'gpu' else 0 
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, use_shared_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, use_shared_memory=True)
    
    model = TransformerOCR(num_classes)
    
    # --- 核心改进：重置训练状态 (TPS 架构变更，不能直接 Load 之前的模型) ---
    RESUME_PATH = 'checkpoints/best_model.pdparams' 
    if os.path.exists(RESUME_PATH):
        print(f"检测到断点，正在加载之前训好的模型: {RESUME_PATH}")
        model.set_state_dict(paddle.load(RESUME_PATH))
    else:
        print("新的 TPS 架构，将从 Epoch 0 开始全新磨合！")

    # --- 核心改进 A: 标签平滑 (Label Smoothing) ---
    # 辅助的 Attention 损失函数使用标签平滑，防止过拟合
    att_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    ctc_loss_fn = nn.CTCLoss(blank=0) 

    # 本地通宵模式：有效 Batch 200 适配 LR 0.0005 建立最强收敛动力
    base_lr = 0.0005 
    steps_per_epoch = len(train_loader)
    # 核心修复：由于梯度累加，调度器真正执行 step 的次数是原来除以 ACCUM_STEPS
    actual_steps_per_epoch = steps_per_epoch // ACCUM_STEPS

    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=EPOCHS * actual_steps_per_epoch)
    # 按照之前的计划，Warmup 设为 10 轮更合理
    lr_scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=lr_scheduler, warmup_steps=10 * actual_steps_per_epoch, start_lr=0, end_lr=base_lr)
    
    opt = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=0.01, grad_clip=nn.ClipGradByNorm(5.0))
    
    # --- 重置 Epoch 计数器 ---
    START_EPOCH = 100 

    # --- 快进学习率调度器 ---
    if START_EPOCH > 0:
        fast_forward_steps = START_EPOCH * actual_steps_per_epoch
        print(f"快进学习率调度器 {fast_forward_steps} 步 (至 Epoch {START_EPOCH})...")
        for _ in range(fast_forward_steps):
            lr_scheduler.step()
        print(f"当前学习率 (Epoch {START_EPOCH} 起点): {lr_scheduler.get_lr():.8f}")
    
    print(f"开始 TPS+双头 全速训练! 设备: {DEVICE}, BS: {BATCH_SIZE}")
    start_time = time.time()
    best_val_acc = 0.0
    
    # --- 强进阶：分段训练 (Multi-stage Training) ---
    # 前 160 轮开启大强度数据增强进行“苦练”
    # 后 40 轮关闭或极端弱化数据增强进行“冲刺” (以求损失函数在干净数据上落到底部)
    
    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        # 200轮架构下：160轮后进入“冲刺期”，关闭复杂的旋转和透视，专注拟合
        if epoch >= 160:
            train_ds.mode = 'none' 
        else:
            train_ds.mode = 'train'

        for i, (imgs, labels, l_lens) in enumerate(train_loader()):
            # 无人房间模式：移除所有休眠，全力冲刺
            # time.sleep(0.15) 
            
            imgs = paddle.to_tensor(imgs)
            labels = paddle.to_tensor(labels)
            l_lens = paddle.to_tensor(l_lens)
            
            # --- 投影前向传播 ---
            # 训练时传入 labels 作为 targets，实现语义辅助学习
            ctc_out, att_out = model(imgs, targets=labels)
            
            # 1. CTC Loss (视觉头)
            ctc_loss = ctc_loss_fn(ctc_out.transpose([1, 0, 2]), labels, paddle.to_tensor([40]*len(imgs), dtype='int64'), l_lens)
            
            # 2. Attention Loss (语义头 - 标签平滑)
            att_loss = att_loss_fn(att_out.reshape([-1, num_classes]), labels.reshape([-1]))
            
            # --- 联合训练：视觉为主 (1.0)，语义为辅 (0.1) ---
            loss = (ctc_loss + 0.1 * att_loss) / ACCUM_STEPS
            
            # 使用标准的 FP32 反向传播，更稳定
            loss.backward()

            # 梯度累加逻辑
            if (i + 1) % ACCUM_STEPS == 0:
                opt.step()
                # 步进学习率并清除梯度
                lr_scheduler.step()
                opt.clear_grad()

            if i % 100 == 0:
                # 修正 ETA 计算：统一使用真实物理加载次数 (steps_per_epoch)，消除跨轮次的时间跳变错觉
                steps_this_session = (epoch - START_EPOCH) * steps_per_epoch + i + 1
                total_steps = (EPOCHS - START_EPOCH) * steps_per_epoch
                remaining_steps = total_steps - steps_this_session
                avg_time_per_step = (time.time() - start_time) / steps_this_session
                eta_seconds = avg_time_per_step * remaining_steps
                
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                current_lr = lr_scheduler.get_lr()
                print(f"Epoch {epoch} Step {i}/{steps_per_epoch}, Loss: {float(loss):.4f}, CTC: {float(ctc_loss):.4f}, ATT: {float(att_loss):.4f}, LR: {current_lr:.8f}, ETA: {eta_str}, {get_gpu_info()}")

        # 验证阶段 (仅使用 CTC 分支进行快速准确识别)
        # 为节约时间，前 180 轮每 5 轮才验证并保存一次；最后 20 轮冲刺期才每轮都验证
        if epoch % 5 != 0 and epoch < 180:
            print(f"--- Epoch {epoch} 跳过验证，直接进入下一轮训练 ---")
            continue

        model.eval()
        total_correct = 0
        total_samples = 0
        with paddle.no_grad():
            for imgs, labels, l_lens in val_loader():
                ctc_out, _ = model(paddle.to_tensor(imgs))
                out_idx = paddle.argmax(ctc_out, axis=-1).numpy()
                for idx in range(len(imgs)):
                    if decode_ctc(out_idx[idx]) == labels[idx][:l_lens[idx]].tolist():
                        total_correct += 1
                    total_samples += 1
        
        val_acc = total_correct / total_samples
        print(f"--- Epoch {epoch} Val Acc: {val_acc:.4%}, Best: {best_val_acc:.4%} ---")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            paddle.save(model.state_dict(), 'checkpoints/best_model.pdparams')
            
        # 每 20 轮强制做一次备份，避免占用过多硬盘空间
        if epoch % 20 == 0:
            paddle.save(model.state_dict(), f'checkpoints/transformer_{epoch}.pdparams')


if __name__ == '__main__':
    train()
