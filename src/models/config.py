"""
模型配置常量.

集中管理类别名、权重路径、默认阈值等.
"""
import os

# 权重路径 (相对于 config.py 所在目录)
_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')

# 类别映射 (class_id → 英文名)
CLASS_NAMES = {
    0: 'danger',
    1: 'electricity',
    2: 'fire',
    3: 'ionizing_radiation',
    4: 'train',
}

# 类别中文名
CLASS_NAMES_ZH = {
    'danger': '注意安全',
    'electricity': '当心触电',
    'fire': '当心火灾',
    'ionizing_radiation': '当心电离辐射',
    'train': '当心火车',
}

# 默认检测阈值
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

# 类别数
NUM_CLASSES = len(CLASS_NAMES)


def get_config():
    """返回完整配置字典."""
    return {
        'weights_path': _WEIGHTS_PATH,
        'class_names': CLASS_NAMES,
        'num_classes': NUM_CLASSES,
        'default_conf': DEFAULT_CONF,
        'default_iou': DEFAULT_IOU,
    }