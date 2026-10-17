import yaml


def load_config(path='config/wav2vec2.yaml'):
    """读取 YAML 配置文件，返回 dict。"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
