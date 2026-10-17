# 语音情感识别 Speech Emotion Recognition

基于 **wav2vec2 预训练模型微调**的语音情感识别项目：输入一段语音，识别其情感类别（愤怒 / 高兴 / 悲伤 / 中性 四类），并提供训练、测试、Web 演示的完整链路。

- **模型**：`facebook/wav2vec2-base`（HuggingFace 预训练语音模型）+ 4 类分类头，PyTorch 实现，支持 GPU 训练
- **数据集**：IEMOCAP，经典 4 类（angry / happy / sad / neutral），**说话人独立**划分
- **指标**：UA（未加权准确率）、WA（加权准确率）、F1（宏平均）、WF1（加权 F1）

---

## 1. 环境

- Python **3.9**（conda 环境 `py39`），GPU 需要 NVIDIA 显卡 + 带 CUDA 的 PyTorch
- 关键依赖：

```bash
conda activate py39
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> GPU 版 torch 会从 PyTorch 官方 CUDA 源下载（约 2.5GB，已在 requirements.txt 里用 `--extra-index-url` 指定）。无显卡则用 CPU 版 torch。

| 包 | 用途 |
| --- | --- |
| `torch`（GPU 版） | 深度学习框架 |
| `transformers` | wav2vec2 预训练模型 |
| `soundfile` / `librosa` | 音频读取与重采样 |
| `scikit-learn` | 评估指标 |
| `flask` | Web 演示后端 |
| `pyyaml` | 读取配置文件 |
| `tqdm` / `matplotlib` | 训练进度条 / 曲线图与混淆矩阵图 |

> 预训练模型需放在本地 `models/wav2vec2-base/`（约 360MB，已加入 `.gitignore`，不会随仓库上传）。首次使用前先下载到该目录，脚本已自动使用国内 `hf-mirror.com` 镜像加速：
>
> ```bash
> huggingface-cli download facebook/wav2vec2-base --local-dir models/wav2vec2-base
> ```
>
> 代码中 `local_files_only=True`（只读本地、不联网），故需先备好模型再运行。

---

## 2. 目录结构

```
SER
├ app.py                      # Flask 演示后端（加载权重，提供 /predict）
├ train.py                    # wav2vec2 微调训练（早停、best/last 权重、训练曲线）
├ test.py                     # 权重测试（UA/WA/F1/WF1 + 混淆矩阵图）
├ prepare.py                  # 原始 IEMOCAP → 4 类 + 说话人独立划分
├ config.py                   # 配置加载函数 load_config()
├ requirements.txt            # 依赖清单（pip install -r requirements.txt）
├ README.md
├ config
│ └ wav2vec2.yaml             # ★ 全局配置：模型、数据路径、超参数、端口
├ datasets
│ ├ IEMOCAP                   # 原始数据集（下载解压后）
│ │ ├ Session1                # 解压后（Session2~5 结构相同）
│ │ │ ├ dialog
│ │ │ │ ├ EmoEvaluation       # ✅ 情感标注 txt（本项目用）
│ │ │ │ │ └ Ses01F_impro01.txt ...
│ │ │ │ ├ wav                 # 整段对话音频（未用）
│ │ │ │ ├ avi / lab / transcriptions
│ │ │ │ └ MOCAP_hand / MOCAP_head / MOCAP_rotated
│ │ │ └ sentences
│ │ │   ├ wav                 # ✅ 切分好的 utterance 音频（本项目用）
│ │ │   │ └ Ses01F_impro01/Ses01F_impro01_F000.wav ...
│ │ │   └ ForcedAlignment / MOCAP_hand / MOCAP_head / MOCAP_rotated
│ │ └ Session2 ~ Session5     # 结构同 Session1
│ └ IEMOCAP_PROCESSED         # 整理后（prepare.py 生成）
│   ├ train                   # Session1-3（3259 条）
│   │ ├ angry
│   │ ├ happy
│   │ ├ sad
│   │ └ neutral
│   ├ val                     # Session4（1031 条）
│   │ ├ angry
│   │ ├ happy
│   │ ├ sad
│   │ └ neutral
│   └ test                    # Session5（1241 条）
│     ├ angry
│     ├ happy
│     ├ sad
│     └ neutral
├ checkpoints
│ ├ WAV2VEC2_IEMOCAP_best.pt  # 最优权重（训练后生成）
│ └ WAV2VEC2_IEMOCAP_last.pt  # 最新权重（训练后生成）
├ img                         # 训练/测试自动生成的图片
│ ├ training_curves.png       # 训练曲线
│ └ confusion_matrix.png      # 混淆矩阵热力图
├ templates
│ └ wav2vec2.html             # 演示页面结构
├ static
│ ├ wav2vec2.css              # 页面样式
│ └ wav2vec2.js               # 交互逻辑（录音/上传/波形/概率）

```

---

## 3. 数据集

IEMOCAP：10 名演员（5 男 5 女）的双人即兴/剧本对话，按 utterance 切分，带情感标注。

| 项 | 内容 |
| --- | --- |
| 情感类别 | angry / happy / sad / neutral（4 类） |
| 划分方式 | **说话人独立**：Session1-3 → train，Session4 → val，Session5 → test |
| 数量 | 共 5531 条（train 3259 / val 1031 / test 1241） |
| 类别分布 | angry 1103 · happy 1636 · sad 1084 · neutral 1708 |

> 原始 IEMOCAP 的情感标签映射为 4 类：`ang→angry`、`hap/exc→happy`（兴奋合并到高兴）、`sad→sad`、`neu→neutral`，其余标签跳过。

---

## 4. 数据准备

把原始 IEMOCAP 整理成训练所需结构：

```bash
python prepare.py                # 生成 datasets/IEMOCAP_PROCESSED
python prepare.py --dry-run      # 只统计，不复制
```

---

## 5. 配置

所有参数和路径集中在 **`config/wav2vec2.yaml`**，训练 / 测试 / 演示三个脚本共用：

```yaml
model_name: models/wav2vec2-base     # 本地预训练模型路径（首次需先下载，见「环境」章节）
data_root: datasets/IEMOCAP_PROCESSED
class_labels: [angry, happy, sad, neutral]
sample_rate: 16000
max_audio_sec: 8
epochs: 10
batch_size: 8
learning_rate: 0.00003   # 即 3e-5（勿写 3e-5，PyYAML 会误解析成字符串）
patience: 3
checkpoint: checkpoints/WAV2VEC2_IEMOCAP_best.pt
port: 5001
```

命令行参数可临时覆盖配置（如 `--epochs 20`、`--batch-size 4`、`--checkpoint ...`）。

---

## 6. 训练

```bash
python train.py
```

- GPU 上每个 epoch 约几分钟（8GB 显存建议 batch 4，batch 8 显存紧张易 OOM）
- 训练时用 tqdm 显示 batch 进度条，每轮打印耗时，结束打印总耗时
- 每个 epoch 打印 `loss / UA / WA / F1 / WF1`，用**验证集（val）**做早停与选最优模型，**测试集（test）只做最终评估**
- 训练结束自动生成训练曲线图 `img/training_curves.png`
- 权重保存为 `checkpoints/WAV2VEC2_IEMOCAP_best.pt`（最优）与 `..._last.pt`（最新）

---

## 7. 测试

```bash
python test.py                                  # 用配置里默认权重（best.pt）
python test.py --checkpoint checkpoints/WAV2VEC2_IEMOCAP_last.pt   # 或用最新权重
```

输出测试集 UA / WA / F1 / WF1 四指标 + 混淆矩阵，并自动保存混淆矩阵热力图到 `img/confusion_matrix.png`。

---

## 8. Web 演示

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:5001/`，支持：

- 实时录音（最长 8 秒）或上传音频文件
- 波形图展示 + 播放
- 四类情感概率可视化

---

## 9. 指标说明

| 指标 | 含义 |
| --- | --- |
| UA | 未加权准确率，各类别召回率平均（不受类别不平衡影响） |
| WA | 加权准确率，即整体正确率 |
| F1 | 宏平均 F1 |
| WF1 | 按类别样本数加权的 F1 |

参考结果（`WAV2VEC2_IEMOCAP_best.pt`，说话人独立）：**UA 62.35% / WA 60.27% / F1 60.96% / WF1 59.97%**。
