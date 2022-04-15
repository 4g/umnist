import random
from pathlib import Path

import cv2
import keras.datasets.mnist

from mnist_gen import UltraMnistGen
from tqdm import tqdm
import numpy as np
from settings import CFG, Paths

def create(img_width, n_samples, net_img_size, location):
  img_mask_gen = UltraMnistGen(out_size=img_width,
                               mask_size=img_width // 2,
                               n_digits=(3, 5),
                               digit_scale=(10 / 4000, 1500 / 4000),
                               batch_size=4, num_batches=50000, output="sum")

  location_dir = Path(location)
  location_dir.mkdir(parents=True, exist_ok=True)
  for digit in range(10):
    digit_dir = location_dir / str(digit)
    digit_dir.mkdir(parents=True, exist_ok=True)

  index = 0
  for i in tqdm(range(n_samples)):
    img, boxes, numbers = img_mask_gen.get_num_image()
    img = (img * 255).astype(np.uint8)
    for number, box in zip(numbers, boxes):
      crop = img[box[0]:box[2], box[1]:box[3]]
      crop = cv2.resize(crop, (net_img_size, net_img_size))
      # cv2.imshow("img", crop)
      # cv2.waitKey(-1)
      path = location_dir / str(number) / f"{index}.jpeg"
      cv2.imwrite(str(path), crop)
      index += 1

  num_samples = 16000
  for digit in range(10):
    digit_dir = location_dir / str(digit)
    files = list(digit_dir.glob("*.jpeg"))
    num_samples = min(len(files), num_samples)

  images_arr = np.zeros((10 * num_samples, net_img_size, net_img_size, 3), dtype=np.uint8)
  labels_arr = np.zeros((10 * num_samples), dtype=np.uint8)
  print(images_arr.shape, labels_arr.shape)

  index = 0
  for digit in range(10):
    digit_dir = location_dir / str(digit)
    files = list(digit_dir.glob("*.jpeg"))
    sample = random.sample(files, num_samples)

    for impath in sample:
      img = cv2.imread(str(impath))
      images_arr[index] = img
      labels_arr[index] = digit
      index += 1

  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  resize_and_color = lambda x: cv2.cvtColor(cv2.resize(x, (net_img_size, net_img_size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

  x_train = [resize_and_color(i) for i in x_train]
  x_test = [resize_and_color(i) for i in x_test]

  x_train = np.concatenate((x_train, x_test, images_arr), axis=0)
  y_train = np.concatenate((y_train, y_test, labels_arr), axis=0)

  print(x_train.shape, y_train.shape)
  print("Creating train test split")
  indices = list(range(len(x_train)))
  random.shuffle(indices)
  test_split = indices[:len(x_train)//10]
  train_split = indices[len(x_train)//10:]
  x_test = x_train[test_split]
  x_train = x_train[train_split]
  y_test = y_train[test_split]
  y_train = y_train[train_split]

  save_dataset(x_train, y_train, x_test, y_test)

def load_dataset():
  path = f"{Paths.base}/mnist_array.npz"
  data = np.load(file=path)
  x_train = data["x_train"]
  y_train = data["y_train"]
  x_test = data["x_test"]
  y_test = data["y_test"]
  return (x_train, y_train), (x_test, y_test)

def save_dataset(x_train, y_train, x_test, y_test):
  path = f"{Paths.base}/mnist_array.npz"
  print("Saving data at", path)
  np.savez_compressed(file=path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", help="data dir")
    args = parser.parse_args()