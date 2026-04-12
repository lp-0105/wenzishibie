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

# --- 1. 参数配置 ---
IMG_H, IMG_W = 48, 160
BATCH_SIZE = 256  # V100 16GB 显存充足，从 160 提升到 256，显著加快单次 Epoch 速度
EPOCHS = 350      # 既然用了更好的卡，增加 Epoch 深度，让模型充分利用语义增强学习
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
            if random.random() > 0.5:
                angle = random.uniform(-5, 5)
                img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
            if random.random() > 0.8:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
            from PIL import ImageEnhance
            if random.random() > 0.7:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.7, 1.3))
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
    paddle.set_device(DEVICE)
    train_ds = CustomOCRDataset('train_data/train.txt', 'train_data', mode='train')
    val_ds = CustomOCRDataset('train_data/val.txt', 'train_data', mode='none')
    # 适配 AI Studio V100 2核 CPU 环境，将 num_workers 调回 4 避免进程切换开销
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    model = TransformerOCR(num_classes)
    
    # --- 核心改进 A: 标签平滑 (Label Smoothing) ---
    # 辅助的 Attention 损失函数使用标签平滑，防止过拟合
    att_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    ctc_loss_fn = nn.CTCLoss(blank=0) 

    base_lr = 0.0003 
    steps_per_epoch = len(train_loader)
    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=EPOCHS * steps_per_epoch)
    lr_scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=lr_scheduler, warmup_steps=20 * steps_per_epoch, start_lr=0, end_lr=base_lr)
    
    opt = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=0.05, grad_clip=nn.ClipGradByNorm(10.0))
    
    print(f"开始语义增强训练! 设备: {DEVICE}, BS: {BATCH_SIZE}")
    start_time = time.time()
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        for i, (imgs, labels, l_lens) in enumerate(train_loader()):
            imgs = paddle.to_tensor(imgs)
            labels = paddle.to_tensor(labels)
            l_lens = paddle.to_tensor(l_lens)
            
            # --- 投影前向传播 ---
            # 训练时传入 labels 作为 targets，实现语义辅助学习
            ctc_out, att_out = model(imgs, targets=labels)
            
            # 1. CTC Loss (视觉头)
            ctc_loss = ctc_loss_fn(ctc_out.transpose([1, 0, 2]), labels, paddle.to_tensor([40]*len(imgs), dtype='int64'), l_lens)
            
            # 2. Attention Loss (语义头 - 标签平滑)
            # targets 右移一位并添加 SOS/Padding 这里的 labels 是已经 padding 过的
            att_loss = att_loss_fn(att_out.reshape([-1, num_classes]), labels.reshape([-1]))
            
            # --- 联合训练：视觉为主 (1.0)，语义为辅 (0.1) ---
            loss = ctc_loss + 0.1 * att_loss
            
            loss.backward()
            opt.step()
            lr_scheduler.step()
            opt.clear_grad()
            
            if i % 100 == 0:
                # 重新计算 ETA 并在日志中显示当前学习率 LR
                processed_steps = epoch * steps_per_epoch + i + 1
                total_steps = EPOCHS * steps_per_epoch
                avg_time_per_step = (time.time() - start_time) / processed_steps
                eta_seconds = avg_time_per_step * (total_steps - processed_steps)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                current_lr = lr_scheduler.get_lr()
                print(f"Epoch {epoch} Step {i}/{steps_per_epoch}, Loss: {float(loss):.4f}, CTC: {float(ctc_loss):.4f}, ATT: {float(att_loss):.4f}, LR: {current_lr:.8f}, ETA: {eta_str}, {get_gpu_info()}")

        # 验证阶段 (仅使用 CTC 分支进行快速准确识别)
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

if __name__ == '__main__':
    train()
