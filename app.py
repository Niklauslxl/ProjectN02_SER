# -*- coding: utf-8 -*-
"""wav2vec2 语音情感识别 Web 演示后端（Flask）。

模型、路径、端口均来自 config/wav2vec2.yaml。

启动:
    python app_wav2vec2.py
访问:
    http://127.0.0.1:5001/
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import io
import numpy as np
import torch
import soundfile as sf
from flask import Flask, request, jsonify, render_template
from transformers import Wav2Vec2ForSequenceClassification

from config import load_config

cfg = load_config()

CLASS_LABELS = cfg['class_labels']
CLASS_LABELS_ZH = cfg['class_labels_zh']
MODEL_NAME = cfg['model_name']
CHECKPOINT = cfg['checkpoint']
SR = cfg['sample_rate']
MAX_LEN = SR * cfg['max_audio_sec']
MIN_LEN = SR // 10  # 最短 0.1 秒
HOST = cfg['host']
PORT = cfg['port']
MAX_UPLOAD_MB = cfg['max_upload_mb']

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True  # 模板修改后自动生效，无需重启

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'加载模型到 {device} ...', flush=True)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CLASS_LABELS), local_files_only=True).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print('模型加载完成', flush=True)


@app.route('/')
def index():
    return render_template('wav2vec2.html', labels=CLASS_LABELS)


@app.route('/predict', methods=['POST'])
def predict():
    f = request.files.get('file')
    if not f:
        return jsonify({'error': '未收到音频文件'}), 400

    try:
        audio, sr = sf.read(io.BytesIO(f.read()))
        if audio.ndim > 1:  # 多声道取平均转单声道
            audio = audio.mean(axis=1)
        if sr != SR:  # 重采样到 16kHz
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        audio = audio.astype(np.float32)
        if len(audio) > MAX_LEN:
            audio = audio[:MAX_LEN]
        if len(audio) < MIN_LEN:  # 太短视为无效
            return jsonify({'error': '音频太短，请至少录 1 秒'}), 400

        duration = len(audio) / SR
        input_values = torch.from_numpy(audio).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_values, dtype=torch.long)

        with torch.no_grad():
            out = model(input_values=input_values, attention_mask=attention_mask)
            probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        return jsonify({
            'labels': CLASS_LABELS,
            'emotion': CLASS_LABELS[pred_idx],
            'emotion_zh': CLASS_LABELS_ZH[CLASS_LABELS[pred_idx]],
            'emotion_idx': pred_idx,
            'probabilities': [float(x) for x in probs],
            'duration': round(duration, 2),
        })
    except Exception as e:
        return jsonify({'error': f'预测失败: {e}'}), 500


if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=False)
