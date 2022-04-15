import time

import cv2
import numpy as np
from mnist_gen import UltraMnistGen
from tqdm import tqdm
from pathlib import Path

def make_dirs(path):
  path.mkdir(exist_ok=True, parents=True)

def create_and_save(img_width, n_samples, prefix, impath, labels_path):
  impath = Path(impath)
  labels_path = Path(labels_path)

  make_dirs(impath)
  make_dirs(labels_path)

  empty = True if prefix == 'empty' else False

  img_mask_gen = UltraMnistGen(out_size=img_width,
                               mask_size=img_width // 2,
                               n_digits=(3, 5),
                               digit_scale=(10 / 4000, 1500 / 4000),
                               batch_size=4,
                               num_batches=50000,
                               output="sum", empty=empty)

  # cv2.namedWindow("image", flags=cv2.WINDOW_NORMAL)
  for i in tqdm(range(n_samples), desc=f"Creating synthetic images {prefix} {impath}"):
    img, boxes, numbers = img_mask_gen.get_num_image()
    img = (img * 255).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(f"{str(impath)}/{prefix}_{i}.jpeg", img)
    # print("index", i)

    labels = []
    for number, box in zip(numbers, boxes):
      x1, y1, x2, y2 = box
      w = x2 - x1
      h = y2 - y1
      cx = (x1 + x2) // 2
      cy = (y1 + y2) // 2
      cx = cx / img_width
      cy = cy / img_width
      w = w / img_width
      h = h / img_width
      label = [cy, cx, h, w]
      labels.append(label)

    label_file = open(labels_path / f"{prefix}_{i}.txt", 'w')
    for number, label in zip(numbers, labels):
      cx, cy, w, h = label
      anno = f"{number} {cx} {cy} {w} {h}\n"
      label_file.write(anno)

    label_file.close()


def make_yaml(base_path, yolo_cfg_path):
  x = f"""path: ../{base_path}  # dataset root dir
train: images/train
val: images/val
test:  # test images (optional)

# Classes
nc: 10  # number of classes
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']\n"""
  f = open(yolo_cfg_path, 'w')
  f.write(x)
  f.close()
