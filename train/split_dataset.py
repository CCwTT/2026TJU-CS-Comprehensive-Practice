"""
数据集划分脚本.

输入: data/scaled/{cat}/*.jpg  +  data/labels/*.txt
输出: data/training/{train|val|test}/{images|labels}/  +  dataset.yaml

划分比例: train 70%, val 20%, test 10% (分层抽样, 按类别独立划分)
"""
import os
import re
import random
import shutil
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 路径配置 (相对于 train/ 目录)
LABELS_DIR = os.path.join('..', 'data', 'labels')
IMAGES_BASE = os.path.join('..', 'data', 'scaled')
OUTPUT_BASE = os.path.join('..', 'data', 'training')
YAML_PATH = os.path.join(OUTPUT_BASE, 'dataset.yaml')

# 类别名 → class_id
CATEGORIES = {
    'danger': 0,
    'electricity': 1,
    'fire': 2,
    'ionizing_radiation': 3,
    'train': 4,
}

# 划分比例
SPLIT_RATIOS = {'train': 0.7, 'val': 0.2, 'test': 0.1}

# 随机种子 (可复现)
RANDOM_SEED = 42


def extract_category(filename: str) -> str | None:
    """
    从文件名中提取类别.
    
    输入: 'danger_ai_001.txt', 'electricity_web_005.txt'
    输出: 'danger', 'electricity'
    """
    for cat in CATEGORIES:
        if filename.startswith(cat + '_'):
            return cat
    return None


def validate_label(txt_path: str) -> list[str] | None:
    """
    校验 YOLO 标注文件.
    
    检查:
      - class_id 合法 (0-4)
      - 坐标在 [0, 1] 内
      - 至少有一个目标
    
    Returns:
        校验通过返回行列表, 失败返回 None
    """
    lines = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
    
    if not lines:
        logger.warning(f'空标注 (无目标): {txt_path}')
        return lines  # 空标注不算致命错误, 但警告
    
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            logger.error(f'格式错误 (需要5个值): {txt_path}:{i+1} → "{line}"')
            return None
        
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            logger.error(f'数值解析失败: {txt_path}:{i+1} → "{line}"')
            return None
        
        if cls_id not in range(len(CATEGORIES)):
            logger.error(f'class_id 越界 ({cls_id}): {txt_path}:{i+1}')
            return None
        
        for name, val in [('cx', cx), ('cy', cy), ('w', w), ('h', h)]:
            if val < 0 or val > 1:
                logger.error(f'bbox {name}={val} 超出 [0,1]: {txt_path}:{i+1}')
                return None
    
    return lines


def find_image(label_filename: str) -> str | None:
    """
    根据标注文件名查找对应的图片.
    
    输入: 'danger_ai_001.txt'
    搜索: data/scaled/danger/danger_ai_001.jpg
    """
    category = extract_category(label_filename)
    if category is None:
        return None
    
    # 去掉 .txt → 加 .jpg
    base = label_filename[:-4]  # 去掉 .txt
    img_path = os.path.join(IMAGES_BASE, category, base + '.jpg')
    
    if os.path.exists(img_path):
        return img_path
    return None


def stratified_split(files_by_category: dict) -> dict:
    """
    分层抽样划分.
    
    Args:
        files_by_category: {category: [(label_path, image_path), ...]}
    
    Returns:
        {split_name: [(label_filename, image_path), ...]}
    """
    random.seed(RANDOM_SEED)
    splits = {name: [] for name in SPLIT_RATIOS}
    
    for cat, pairs in files_by_category.items():
        # 打乱
        shuffled = list(pairs)
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = int(n * SPLIT_RATIOS['train'])
        n_val = int(n * SPLIT_RATIOS['val'])
        # test 拿剩余的
        
        splits['train'].extend(shuffled[:n_train])
        splits['val'].extend(shuffled[n_train:n_train + n_val])
        splits['test'].extend(shuffled[n_train + n_val:])
        
        logger.info(
            f'  {cat}: total={n}, '
            f'train={n_train}, val={n_val}, test={n - n_train - n_val}'
        )
    
    return splits


def copy_files(splits: dict) -> dict:
    """
    复制图片和标注到目标目录.
    
    Returns:
        {split_name: count}
    """
    counts = defaultdict(int)
    
    for split_name, pairs in splits.items():
        img_dir = os.path.join(OUTPUT_BASE, split_name, 'images')
        lbl_dir = os.path.join(OUTPUT_BASE, split_name, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        for lbl_path, img_path in pairs:
            # 文件名 (不含路径)
            base = os.path.basename(lbl_path)
            fname = base[:-4]  # 去掉 .txt
            
            # 复制图片
            dst_img = os.path.join(img_dir, fname + '.jpg')
            shutil.copy2(img_path, dst_img)
            
            # 复制标注
            dst_lbl = os.path.join(lbl_dir, base)
            shutil.copy2(lbl_path, dst_lbl)
            
            counts[split_name] += 1
    
    return counts


def generate_yaml():
    """生成 dataset.yaml."""
    # 相对于 yaml 所在目录 (data/training/) 的路径
    content = f"""# 危险标志检测数据集配置
# 自动生成, 请勿手动修改

path: .
train: train/images
val: val/images
test: test/images

nc: {len(CATEGORIES)}
names:
"""
    for cat, cid in sorted(CATEGORIES.items(), key=lambda x: x[1]):
        content += f"  {cid}: {cat}\n"
    
    os.makedirs(os.path.dirname(os.path.abspath(YAML_PATH)), exist_ok=True)
    with open(YAML_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f'dataset.yaml 已生成: {YAML_PATH}')


def main():
    # 1. 扫描所有标注文件
    if not os.path.isdir(LABELS_DIR):
        logger.error(f'标注目录不存在: {LABELS_DIR}')
        return
    
    txt_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]
    logger.info(f'找到 {len(txt_files)} 个标注文件')
    
    # 2. 校验 + 匹配图片, 按类别分组
    files_by_category = defaultdict(list)
    invalid_count = 0
    
    for txt_file in sorted(txt_files):
        txt_path = os.path.join(LABELS_DIR, txt_file)
        
        # 校验标注
        if validate_label(txt_path) is None:
            invalid_count += 1
            continue
        
        # 查找图片
        img_path = find_image(txt_file)
        if img_path is None:
            logger.error(f'找不到对应图片: {txt_file}')
            invalid_count += 1
            continue
        
        # 归类
        cat = extract_category(txt_file)
        if cat is None:
            logger.error(f'无法识别类别: {txt_file}')
            invalid_count += 1
            continue
        
        files_by_category[cat].append((txt_path, img_path))
    
    logger.info(f'有效配对: {sum(len(v) for v in files_by_category.values())}, '
                f'无效: {invalid_count}')
    
    # 3. 分层抽样
    logger.info('开始划分数据集 (7:2:1)...')
    splits = stratified_split(files_by_category)
    
    # 4. 复制文件
    logger.info('复制文件到 data/training/...')
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    counts = copy_files(splits)
    
    # 5. 生成配置
    generate_yaml()
    
    # 6. 打印汇总
    print('\n' + '=' * 50)
    print(f'  {"数据集划分完成":^46}')
    print('=' * 50)
    total = 0
    for split_name in ['train', 'val', 'test']:
        c = counts.get(split_name, 0)
        total += c
        print(f'  {split_name:<8}: {c:>4} 张')
    print(f'  {"合计":<8}: {total:>4} 张')
    print('=' * 50 + '\n')


if __name__ == '__main__':
    main()