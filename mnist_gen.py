from tensorflow import keras
import random
import cv2

import geomlib
import numpy as np
from tqdm import tqdm
from checkerboard import make_checkerboard
import albumentations as A
from settings import CFG, Paths

class MnistGen:
  def __init__(self):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    self.images = np.concatenate((x_train, x_test))
    self.labels = np.concatenate((y_train, y_test))
    self.indices = list(range(len(self.images)))
    self.img_size = self.images.shape[2]
    self.img_by_labels = {}
    self.label_index = {i:0 for i in range(10)}
    self.prep_label_dict()

  def prep_label_dict(self):
    labels = np.unique(self.labels)
    for label in labels:
      self.img_by_labels[label] = self.images[self.labels == label]

  def get_random_image(self, label):
    images = self.img_by_labels[label]
    index = self.label_index[label]
    image = images[index]
    self.label_index[label] = (self.label_index[label] + 1) % len(images)
    return image


class ComboGen(keras.utils.Sequence):
  def __init__(self, gen1, gen2):
    # combines two generators
    # make sure both generators have same number of batches
    self.gen1 = gen1
    self.gen2 = gen2

  def __len__(self):
    # both have same length
    return len(self.gen1)

  def __getitem__(self, item):
    x1, y1 = self.gen1[item]
    x2, y2 = self.gen2[item]
    x3 = np.concatenate([x1, x2])
    y3 = np.concatenate([y1, y2])
    return x3, y3

  def on_epoch_end(self):
    self.gen1.on_epoch_end()
    self.gen2.on_epoch_end()

class CompGen(keras.utils.Sequence):
  def __init__(self, batch_size, out_size, split, augment=False, shuffle=True):
    """
    :param batch_size:
    :param outsize: image output size
    :param num_batches: number of batches in the dataset
    :param split: train or test
    """
    self.split = split
    self.augment = augment
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.out_size = out_size
    self.load()
    self.imgb = np.zeros((self.batch_size, self.out_size, self.out_size, 3), dtype=np.float32)
    self.maskb = np.zeros((self.batch_size, 1), dtype=np.float32)
    self.transform = A.Compose([
            A.InvertImg(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_NEAREST),
            A.Perspective(p=0.1, interpolation=cv2.INTER_NEAREST),
        ], p=.5)

  def __len__(self):
    return len(self.data) // self.batch_size

  def load(self):
    data = np.load(file=f"data/{self.split}.cache.npz")
    images = data["images"]
    counts = data["counts"]
    self.images = images
    self.counts = counts
    self.data = list(range(len(self.images)))

  def __getitem__(self, item):
    batch = self.data[item*self.batch_size:(item + 1)*self.batch_size]
    for idx, dindex in enumerate(batch):
      image = self.images[dindex]
      count = self.counts[dindex]
      image = np.unpackbits(image, axis=0)
      image = image.astype(np.uint8) * 255

      if self.augment:
        image = self.transform(image=image)["image"]
      image = UltraMnistGen.preprocess_image(image)
      self.imgb[idx,:,:, 0] = image
      self.imgb[idx, :, :, 1] = image
      self.imgb[idx, :, :, 2] = image

      self.maskb[idx] = count

    return self.imgb, self.maskb

  def on_epoch_end(self):
    if self.shuffle:
      random.shuffle(self.data)

class UltraMnistGen(keras.utils.Sequence):
  def __init__(self, out_size,mask_size,n_digits, digit_scale, batch_size, num_batches, nchannels=10, output='mask', empty=False):
    # possible values of output = "mask" or "sum"
    self.empty = empty
    self.mnist_gen = MnistGen()
    self.out_size = out_size
    self.mask_size = mask_size
    self.n_digits = n_digits
    self.digit_scale = digit_scale
    self.batch_size = batch_size
    self.disc_size = max(int(max(np.sqrt(self.out_size), 1)), 2)
    self.gaussian = self.get_gaussian(self.disc_size * 2 + 1)
    self.num_batches = num_batches
    self.output = output
    self.keypoint_mask_orig = np.zeros((self.out_size, self.out_size, nchannels), dtype=np.float32)
    self.locgen = geomlib.CachedLocationGenerator(size=self.out_size,
                                                  ratio=20,
                                                  minsize=0.03,
                                                  maxsize=0.5,
                                                  iters=30000)
    self.locgen.load(Paths.locgen)
    self.imgb = np.zeros((self.batch_size, self.out_size, self.out_size), dtype=np.float32)
    self.digits_selected = {i:0 for i in range(10)}

    if self.output == 'mask':
      self.maskb = np.zeros((self.batch_size, self.mask_size, self.mask_size, 10), dtype=np.float32)
    elif self.output == 'sum':
      self.maskb = np.zeros((self.batch_size, 1), dtype=np.float32)



  def __len__(self):
    return self.num_batches

  def get_numbers(self, n_digits):
    selected = []
    for i in range(n_digits):
      min_selected = sorted(self.digits_selected, key=self.digits_selected.get)[0]
      selected.append(min_selected)
      self.digits_selected[min_selected] += 1
    return selected

  def get_gaussian(self, size):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    sigma, mu = .5, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    g = (g - np.min(g)) / (np.max(g) - np.min(g))
    return g

  def get_num_image(self):
    n_digits = random.randint(self.n_digits[0], self.n_digits[1])
    numbers = self.get_numbers(n_digits)
    if self.empty:
      numbers = []
    background = np.zeros((self.out_size, self.out_size), dtype=np.uint8)
    # locgen = geomlib.LocationGenerator(size=self.out_size, ratio=50, minsize=self.digit_scale[0], maxsize=self.digit_scale[1])
    boxes = []
    self.locgen.reset()
    size_min = int(self.digit_scale[0] * self.out_size)
    size_min = max(size_min, 2)
    size_max = int(self.digit_scale[1] * self.out_size)

    for number in numbers:
      # print("Number", number)
      image = self.mnist_gen.get_random_image(number)
      box = self.locgen.insert()
      box_ratio = self.out_size / CFG.locgen_img_size
      box = [int(b*box_ratio) for b in box]
      x1, y1, x2, y2 = box
      square_box = [x1, y1, x2, y1+x2-x1]
      box = square_box
      x1, y1, x2, y2 = box
      new_size = x2 - x1
      size_min_ = random.randint(min(size_min, new_size), new_size)
      if size_min_ > 20:
        inter = cv2.INTER_NEAREST
      else:
        inter = cv2.INTER_CUBIC

      resized_image = cv2.resize(image, dsize=(size_min_, size_min_), interpolation=inter)
      left_border = (new_size - size_min_)//2
      right_border = (new_size - size_min_) - left_border
      new_box = (box[0] + left_border, box[1] + left_border, box[2] - left_border, box[3] - left_border)
      boxes.append(new_box)
      resized_image = cv2.copyMakeBorder(resized_image,
                                         top=left_border,
                                         bottom=right_border,
                                         left=left_border,
                                         right=right_border,
                                         borderType=cv2.BORDER_CONSTANT)
      # random_thresh = random.randint(180, 220)
      # random_thresh = 180
      # resized_image[resized_image > random_thresh] = 255
      # resized_image[resized_image < 255] = 0
      # if size_min_ > 30:
        # print("Threshold applied")
      _, resized_image = cv2.threshold(resized_image, thresh=160, maxval=255, type=cv2.THRESH_OTSU)
      # cv2.imshow("res", resized_image)
      # cv2.waitKey(-1)
      background[box[0]:box[2], box[1]:box[3]] = resized_image


    checkerboard = make_checkerboard(self.out_size)
    background = cv2.bitwise_xor(checkerboard, background)
    edge = self.preprocess_image(background)
    return edge, boxes, numbers

  def get_mask(self, boxes, numbers):
    keypoint_mask = self.keypoint_mask_orig.copy()
    for number, box in zip(numbers, boxes):
      box_x = (box[1] + box[3]) // 2
      box_y = (box[0] + box[2]) // 2

      left_space_x = min(box_x, self.disc_size)
      left_space_y = min(box_y, self.disc_size)

      right_space_x = min(self.out_size - box_x, self.disc_size + 1)
      right_space_y = min(self.out_size - box_y, self.disc_size + 1)

      keypoint_mask[box_y - left_space_y: box_y + right_space_y,
      box_x - left_space_x: box_x + right_space_x, number] = self.gaussian[self.disc_size - left_space_y:self.disc_size + right_space_y,
               self.disc_size - left_space_x:self.disc_size + right_space_x]

    keypoint_mask = cv2.resize(keypoint_mask, dsize=(self.mask_size, self.mask_size))
    return keypoint_mask

  def __getitem__(self, idx):
    # Read filenames and fill the batch
    for index in range(self.batch_size):
      img, boxes, numbers = self.get_num_image()
      if self.output == 'mask':
        mask = self.get_mask(boxes, numbers)
      else:
        mask = sum(numbers)

      self.imgb[index] = img
      self.maskb[index] = mask

    return self.imgb, self.maskb

  @staticmethod
  def preprocess_image(img):
    img = img.astype(np.float32)
    img = img / 127.5 - 1.
    return img

  def on_epoch_end(self):
    self.locgen.shuffle()

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  img_size = 2560
  # # from tqdm import tqdm
  # # for i in tqdm(range(100)):
  # #   x = img_mask_gen[i]
  #
  # imgb, maskb = img_mask_gen[0]
  # for img, mask in zip(imgb, maskb):
  #   mask = cv2.resize(mask, (img.shape[0], img.shape[1]))
  #   mask = np.max(mask, axis=-1)
  #   img = np.max(img, axis=-1)
  #   ans = cv2.max(mask, img)
  #   cv2.imshow("image", img)
  #   cv2.waitKey(-1)

  # img_mask_gen2 = CompGen(batch_size=8, out_size=img_size, split=f"train_1_{img_size}", augment=True)
  img_mask_gen1 = UltraMnistGen(out_size=img_size,
                               mask_size=img_size//4,
                               n_digits=(3, 5),
                               digit_scale=(10/4000, 1500/4000),
                               batch_size=1, num_batches=1000, output='sum')

  #
  # imgb, maskb = img_mask_gen2[0]
  # for img in imgb:
  #   img = img * 255
  #   img = img.astype(np.uint8)
  #   cv2.imshow("real", img)
  #   cv2.waitKey(-1)

  cv2.namedWindow("fake", flags=cv2.WINDOW_NORMAL)
  for i in tqdm(range(10000)):
    imgb, maskb = img_mask_gen1[i]
    img = imgb[0]
    img = img * 255
    img = img.astype(np.uint8)
    # cv2.imshow("fake", img)
    # cv2.waitKey(-1)


  #
  # img_mask_gen3 = ComboGen(img_mask_gen1, img_mask_gen2)
  # imgb, maskb = img_mask_gen3[0]
  # for img, mask in zip(imgb, maskb):
  #   img = img * 255
  #   img = img.astype(np.uint8)
  #   print(mask)
  #   cv2.imshow("combo", img)
  #   cv2.waitKey(-1)
  #
