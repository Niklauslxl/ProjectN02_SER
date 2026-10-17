import os
import re
import glob
import shutil
import argparse
from collections import defaultdict

TRAIN_SESSIONS = ['Session1', 'Session2', 'Session3']
VAL_SESSIONS = ['Session4']
TEST_SESSIONS = ['Session5']

# IEMOCAP 原始情感标签 → 4 类
EMOTION_MAP = {
    'ang': 'angry',
    'hap': 'happy',
    'exc': 'happy',   # 兴奋合并到高兴
    'sad': 'sad',
    'neu': 'neutral',
}

# 标注行: [6.2901 - 8.2357]  Ses01F_impro01_F000  neu  [2.5000, 2.5000, 2.5000]
LABEL_LINE = re.compile(r'^\[[\d.]+ - [\d.]+\]\s+(\S+)\s+(\S+)\s+\[')
# 说话人前缀: Ses01F / Ses01M ...
SPEAKER = re.compile(r'^(Ses\d{2}[MF])')


def parse_emotion_labels(src_root, session):
    """解析某个 Session 的标注，返回 {utterance_name: 原始情感标签}。"""
    labels = {}
    eval_dir = os.path.join(src_root, session, 'dialog', 'EmoEvaluation')
    for txt in glob.glob(os.path.join(eval_dir, '*.txt')):
        with open(txt, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = LABEL_LINE.match(line.strip())
                if m:
                    labels[m.group(1)] = m.group(2)
    return labels


def collect(session, split, labels):
    """收集一个 Session 的 (src_wav, dst_relpath)，返回列表；跳过非 4 类情感。"""
    src_root = os.path.join(ARGS.src, session)
    wavs = sorted(glob.glob(os.path.join(src_root, 'sentences', 'wav', '*', '*.wav')))
    out = []
    for wav in wavs:
        utt = os.path.splitext(os.path.basename(wav))[0]
        emo = labels.get(utt)
        if emo not in EMOTION_MAP:
            continue
        target = EMOTION_MAP[emo]
        speaker = SPEAKER.match(utt)
        speaker = speaker.group(1) if speaker else 'unknown'
        dst_name = f'{utt}-{target}-{speaker}.wav'
        dst_rel = os.path.join(split, target, dst_name)
        out.append((wav, dst_rel))
    return out


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='datasets/IEMOCAP')
    parser.add_argument('--dst', type=str, default='datasets/IEMOCAP_PROCESSED')
    parser.add_argument('--dry-run', action='store_true', help='只统计不复制')
    ARGS = parser.parse_args()

    counter = defaultdict(int)
    for session in TRAIN_SESSIONS + VAL_SESSIONS + TEST_SESSIONS:
        if session in TRAIN_SESSIONS:
            split = 'train'
        elif session in VAL_SESSIONS:
            split = 'val'
        else:
            split = 'test'
        labels = parse_emotion_labels(ARGS.src, session)
        pairs = collect(session, split, labels)
        print(f'{session} → {split}: 标注 {len(labels)} 条, 4 类有效 {len(pairs)} 条')
        for wav, dst_rel in pairs:
            counter[dst_rel.split(os.sep)[1]] += 1
            if ARGS.dry_run:
                continue
            dst_path = os.path.join(ARGS.dst, dst_rel)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if not os.path.exists(dst_path):
                shutil.copy2(wav, dst_path)

    print('\n=== 各类别统计 ===')
    for k in ['angry', 'happy', 'sad', 'neutral']:
        print(f'{k:<8} {counter[k]}')
    print(f'合计     {sum(counter.values())}')
    if ARGS.dry_run:
        print('\n[dry-run] 未复制任何文件。去掉 --dry-run 即可实际生成。')


if __name__ == '__main__':
    main()
