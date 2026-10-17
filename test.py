import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import glob
import argparse
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from transformers import Wav2Vec2ForSequenceClassification
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境，直接存图不弹窗
import matplotlib.pyplot as plt

from config import load_config

class SERDataset(Dataset):
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
    audios = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    max_len = max(len(a) for a in audios)
    input_values = torch.zeros(len(audios), max_len)
    attention_mask = torch.zeros(len(audios), max_len, dtype=torch.long)
    for i, a in enumerate(audios):
        input_values[i, :len(a)] = torch.from_numpy(a)
        attention_mask[i, :len(a)] = 1
    return input_values, attention_mask, torch.tensor(labels, dtype=torch.long)

def plot_confusion_matrix(cm, class_labels, save_path):
    """绘制并保存混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap='Blues')
    # 每个格子标注数值
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=12)
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'混淆矩阵图已保存: {save_path}', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/wav2vec2.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_name = cfg['model_name']
    class_labels = cfg['class_labels']
    data_root = cfg['data_root']
    sample_rate = cfg['sample_rate']
    max_len = sample_rate * cfg['max_audio_sec']
    checkpoint = args.checkpoint or cfg['checkpoint']
    batch_size = args.batch_size if args.batch_size is not None else cfg['test_batch_size']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print(f'权重: {checkpoint}')

    test_ds = SERDataset('test', data_root, class_labels, sample_rate, max_len)
    print(f'test {len(test_ds)}')
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name, num_labels=len(class_labels), local_files_only=True).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_idx, (input_values, attention_mask, labels) in enumerate(test_loader):
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            out = model(input_values=input_values, attention_mask=attention_mask)
            all_pred.append(out.logits.argmax(-1).cpu().numpy())
            all_true.append(labels.numpy())
            if (batch_idx + 1) % 50 == 0:
                print(f'  batch {batch_idx + 1}/{len(test_loader)}', flush=True)

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    ua = balanced_accuracy_score(true, pred)
    wa = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro')
    wf1 = f1_score(true, pred, average='weighted')

    print('\n=== 测试集指标 ===')
    print(f'UA  {ua:.4f}')
    print(f'WA  {wa:.4f}')
    print(f'F1  {f1:.4f}')
    print(f'WF1 {wf1:.4f}')

    cm = confusion_matrix(true, pred, labels=range(len(class_labels)))
    print('\n=== 混淆矩阵（行=真实，列=预测）===')
    header = ' ' * 10 + ' '.join(f'{l:<8}' for l in class_labels)
    print(header)
    for i, row in enumerate(cm):
        print(f'{class_labels[i]:<10}' + ' '.join(f'{v:<8}' for v in row))

    # 保存混淆矩阵图
    os.makedirs('img', exist_ok=True)
    plot_confusion_matrix(cm, class_labels, 'img/confusion_matrix.png')

if __name__ == '__main__':
    main()
