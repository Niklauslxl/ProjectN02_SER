import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 自动使用国内镜像下载/加载 HuggingFace 模型（避免连 huggingface.co 超时）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import glob
import argparse
import time
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from transformers import Wav2Vec2ForSequenceClassification
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境，直接存图不弹窗
import matplotlib.pyplot as plt

from config import load_config

class SERDataset(Dataset):
    """从 wav 加载原始波形与标签."""

    def __init__(self, split, data_root, class_labels, sample_rate, max_len):
        self.samples = []
        self.labels = []
        base = os.path.join(data_root, split)
        for label in class_labels:
            for wav in sorted(glob.glob(os.path.join(base, label, '*.wav'))):
                self.samples.append(wav)
                self.labels.append(class_labels.index(label))
        self.sample_rate = sample_rate
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio, sr = sf.read(self.samples[idx])
        if sr != self.sample_rate:
            raise ValueError(f'采样率 {sr} 不符合，需要 {self.sample_rate}')
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]
        return audio.astype(np.float32), self.labels[idx]

def collate_fn(batch):
    """动态 padding 到 batch 内最长，生成 attention_mask."""
    audios = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    max_len = max(len(a) for a in audios)
    input_values = torch.zeros(len(audios), max_len)
    attention_mask = torch.zeros(len(audios), max_len, dtype=torch.long)
    for i, a in enumerate(audios):
        input_values[i, :len(a)] = torch.from_numpy(a)
        attention_mask[i, :len(a)] = 1
    return input_values, attention_mask, torch.tensor(labels, dtype=torch.long)

def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for input_values, attention_mask, labels in loader:
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            out = model(input_values=input_values, attention_mask=attention_mask,
                        labels=labels)
            total_loss += out.loss.item() * input_values.size(0)
            n += input_values.size(0)
            all_pred.append(out.logits.argmax(-1).cpu().numpy())
            all_true.append(labels.cpu().numpy())
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    return (total_loss / n,
            balanced_accuracy_score(true, pred),
            accuracy_score(true, pred),
            f1_score(true, pred, average='macro'),
            f1_score(true, pred, average='weighted'))

def plot_training_curves(history, save_path):
    """根据每个 epoch 的历史指标绘制训练曲线（左 loss / 右四指标）。"""
    epochs = range(1, len(history['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # 左图：训练 loss + 验证 loss
    axes[0].plot(epochs, history['loss'], 'o-', color='#0f766e', lw=2, label='train loss')
    axes[0].plot(epochs, history['val_loss'], 'o-', color='#c07a11', lw=2, label='val loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 右图：UA / WA / F1 / WF1 四指标
    metrics = [('ua', '#0f766e', 'UA'), ('wa', '#c07a11', 'WA'),
               ('f1', '#2458c5', 'F1'), ('wf1', '#5b6470', 'WF1')]
    for key, color, name in metrics:
        axes[1].plot(epochs, history[key], 'o-', color=color, lw=2, label=name)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'训练曲线已保存: {save_path}', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/wav2vec2.yaml')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 命令行参数优先，未指定则用配置文件
    model_name = args.model or cfg['model_name']
    epochs = args.epochs if args.epochs is not None else cfg['epochs']
    batch_size = args.batch_size if args.batch_size is not None else cfg['batch_size']
    lr = float(args.lr if args.lr is not None else cfg['learning_rate'])
    patience = args.patience if args.patience is not None else cfg['patience']

    class_labels = cfg['class_labels']
    data_root = cfg['data_root']
    sample_rate = cfg['sample_rate']
    max_len = sample_rate * cfg['max_audio_sec']
    checkpoint_dir = cfg['checkpoint_dir']
    checkpoint_name = cfg['checkpoint_name']
    grad_clip = cfg.get('grad_clip', 1.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}'
          + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))
    print(f'模型: {model_name}')

    train_ds = SERDataset('train', data_root, class_labels, sample_rate, max_len)
    val_ds = SERDataset('val', data_root, class_labels, sample_rate, max_len)
    test_ds = SERDataset('test', data_root, class_labels, sample_rate, max_len)
    print(f'train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name, num_labels=len(class_labels),
        local_files_only=True).to(device)  #首次运行写False，之后运行改成True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_wa = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'loss': [], 'val_loss': [], 'ua': [], 'wa': [], 'f1': [], 'wf1': []}

    train_start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', ncols=100)
        for input_values, attention_mask, labels in pbar:
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(input_values=input_values,
                        attention_mask=attention_mask,
                        labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += out.loss.item() * input_values.size(0)
            pbar.set_postfix({'loss': f'{out.loss.item():.4f}'})

        val_loss, ua, wa, f1, wf1 = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_ds)
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch:3d}/{epochs} | 耗时 {epoch_time:5.1f}s | loss {avg_loss:.4f} | '
              f'val_loss {val_loss:.4f} | val UA {ua:.4f} | WA {wa:.4f} | F1 {f1:.4f} | WF1 {wf1:.4f}', flush=True)

        history['loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['ua'].append(ua)
        history['wa'].append(wa)
        history['f1'].append(f1)
        history['wf1'].append(wf1)

        # 保存最新权重（每个 epoch 都覆盖 last.pt）
        torch.save(model.state_dict(),
                   os.path.join(checkpoint_dir, f'{checkpoint_name}_last.pt'))

        if wa > best_wa:
            best_wa = wa
            best_epoch = epoch
            patience_counter = 0
            # 保存最优权重（刷新最佳时覆盖 best.pt）
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, f'{checkpoint_name}_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'早停于 epoch {epoch}（最佳 epoch {best_epoch}，WA {best_wa:.4f}）')
                break

    model.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, f'{checkpoint_name}_best.pt')))
    _, ua, wa, f1, wf1 = evaluate(model, test_loader, device)
    print('\n=== 最终测试集指标（最佳模型）===')
    print(f'UA  {ua:.4f}')
    print(f'WA  {wa:.4f}')
    print(f'F1  {f1:.4f}')
    print(f'WF1 {wf1:.4f}')

    total_time = time.time() - train_start
    print(f'\n训练总耗时: {total_time / 60:.1f} 分钟 ({total_time:.0f} 秒)', flush=True)

    # 训练结束，生成训练曲线图（保存到 img/）
    os.makedirs('img', exist_ok=True)
    plot_training_curves(history, 'img/training_curves.png')

if __name__ == '__main__':
    main()
