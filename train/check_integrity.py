"""检查 data/training/ 完整性: 图片可读 + 标注存在"""
from pathlib import Path
import cv2

training = Path('../data/training')
issues = []
total = 0

for split in ['train', 'val', 'test']:
    img_dir = training / split / 'images'
    lbl_dir = training / split / 'labels'
    for img_path in sorted(img_dir.glob('*.jpg')):
        total += 1
        img = cv2.imread(str(img_path))
        if img is None:
            issues.append(f'CORRUPT: {img_path}')
        lbl = lbl_dir / (img_path.stem + '.txt')
        if not lbl.exists():
            issues.append(f'MISSING_LABEL: {img_path}')

print(f'Scanned {total} images')
if issues:
    print(f'{len(issues)} ISSUES:')
    for i in issues:
        print(f'  {i}')
else:
    print('All images & labels OK')