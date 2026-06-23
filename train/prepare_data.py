import os
import re
import csv
import logging
import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] %(message)s'
)
logger = logging.getLogger(__name__)

SIZE = (640, 640)
INPUT_DIR = os.path.join('..', 'data', 'raw')
OUTPUT_DIR = os.path.join('..', 'data', 'scaled')

# 图片文件扩展名
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

# 标签类名映射
# 需要与后续训练配置一致

def letterbox_resize(img: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    等比例缩放 + 黑边填充到 target_size × target_size.
    
    Args:
        img: BGR 图像 (H, W, 3)
        target_size: (W, H) 目标尺寸
    
    Returns:
        缩放并填充后的图像
    """
    h, w = img.shape[:2]
    tw, th = target_size
    # 计算缩放比例 (按长边缩放)
    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放 (INTER_AREA 适合缩小)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 创建黑底画布并居中放置
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x_offset = (tw - new_w) // 2
    y_offset = (th - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def process_image(input_path: str, target_size: tuple) -> np.ndarray | None:
    """
    读取图片、去 alpha 通道、letterbox 缩放.
    
    Args:
        input_path: 图片路径
        target_size: 目标尺寸
    
    Returns:
        处理后的 BGR 图像, 失败返回 None
    """
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.error(f'读取失败: {input_path}')
        return None
    
    # 处理 alpha 通道 (BGRA → BGR)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # letterbox 缩放
    img = letterbox_resize(img, target_size)
    return img


def parse_filename(filename: str) -> tuple[str, str] | None:
    """
    解析文件名, 提取来源和序号.
    
    输入: 'ai (1).png', 'web (12).jpeg'
    输出: ('ai', '001'), ('web', '012')
    
    Returns:
        (source, padded_index) 或 None (解析失败)
    """
    # 提取来源前缀
    if filename.startswith('ai'):
        source = 'ai'
    elif filename.startswith('web'):
        source = 'web'
    else:
        return None
    
    # 提取括号中的数字
    match = re.search(r'\((\d+)\)', filename)
    if not match:
        return None
    
    index = int(match.group(1))
    padded_index = f'{index:03d}'
    return source, padded_index


def print_stats(info: dict, output_csv: str) -> None:
    """打印统计表格到终端并写入 CSV."""
    print('\n' + '=' * 60)
    print(f'{"类别":<22} {"总数":>6}  {"AI生成":>6}  {"网图":>6}  {"AI比例":>7}')
    print('-' * 60)
    
    total_all = 0
    ai_all = 0
    web_all = 0
    
    rows = []
    for category in sorted(info.keys()):
        stats = info[category]
        ai = stats.get('ai', 0)
        web = stats.get('web', 0)
        total = stats.get('total', 0)
        pct = f'{ai/total*100:.1f}%' if total > 0 else 'N/A'
        print(f'{category:<22} {total:>6}  {ai:>6}  {web:>6}  {pct:>7}')
        rows.append({'类别': category, '总数': total, 'AI生成': ai, '网图': web, 'AI比例': pct})
        total_all += total
        ai_all += ai
        web_all += web
    
    print('-' * 60)
    pct_all = f'{ai_all/total_all*100:.1f}%' if total_all > 0 else 'N/A'
    print(f'{"合计":<22} {total_all:>6}  {ai_all:>6}  {web_all:>6}  {pct_all:>7}')
    print('=' * 60 + '\n')
    
    # 写入 CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['类别', '总数', 'AI生成', '网图', 'AI比例'])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({'类别': '合计', '总数': total_all, 'AI生成': ai_all, '网图': web_all, 'AI比例': pct_all})
    logger.info(f'统计结果已写入 {output_csv}')


def main():
    # 统计信息: {类别: {"ai": N, "web": N, "total": N}}
    info = {}
    
    # 获取所有类别目录
    categories = [
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ]
    
    if not categories:
        logger.error(f'未找到类别目录, INPUT_DIR={INPUT_DIR}')
        return
    
    logger.info(f'找到 {len(categories)} 个类别: {categories}')
    
    for category in categories:
        category_dir = os.path.join(INPUT_DIR, category)
        # 输出子目录
        out_category_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(out_category_dir, exist_ok=True)
        
        logger.info(f'处理类别: {category}')
        info.setdefault(category, {'ai': 0, 'web': 0, 'total': 0})
        
        # 收集该类下的所有图片
        files = [
            f for f in os.listdir(category_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ]
        
        for filename in tqdm(files, desc=f'  {category}', unit='img'):
            # 解析文件名
            parsed = parse_filename(filename)
            if parsed is None:
                logger.warning(f'无法解析文件名: {filename}, 跳过')
                continue
            source, idx = parsed
            
            # 读取并处理
            input_path = os.path.join(category_dir, filename)
            img = process_image(input_path, SIZE)
            if img is None:
                continue
            
            # 输出: data/scaled/{category}/{category}_{source}_{idx}.jpg
            output_name = f'{category}_{source}_{idx}.jpg'
            output_path = os.path.join(out_category_dir, output_name)
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # 统计
            info[category][source] = info[category].get(source, 0) + 1
            info[category]['total'] += 1
    
    # 输出统计
    csv_path = os.path.join(OUTPUT_DIR, 'stats.csv')
    print_stats(info, csv_path)
    logger.info('预处理完成!')

if __name__ == '__main__':
    main()
