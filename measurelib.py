import copy
import json

import cv2
import numpy as np
from tensorflow import  keras
from tqdm import tqdm
import os
import logging
from pathlib import Path
from yololib import intersection_over_minArea
from yololib import draw_label_on_img

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

class DigitGen(keras.utils.Sequence):
  def __init__(self, images, batch_size, out_size):
    self.images = images
    self.batch_size = batch_size
    self.data = list(range(len(self.images)))
    self.out_size = out_size
    self.imgb = np.zeros((self.batch_size, self.out_size, self.out_size, 3), dtype=np.float32)

  def __len__(self):
    return len(self.data) // self.batch_size + 1

  def __getitem__(self, item):
    batch = self.data[item*self.batch_size:(item + 1)*self.batch_size]
    for idx, dindex in enumerate(batch):
      image = self.images[dindex]
      image = cv2.resize(image, dsize=(self.out_size, self.out_size))
      self.imgb[idx,:,:] = image.astype(np.float32) / 255

    return self.imgb


def load_data(split):
  lines = [x.strip().split(",") for x in open(f"data/{split}.csv")]
  data = {im_id: int(s) for im_id, s in lines[1:]}
  return data


def store_digit_images(labels_dict, store, img_source):
  store = Path(store)
  store.mkdir(parents=True, exist_ok=True)

  id_sorted = sorted(labels_dict.keys())

  for id in tqdm(id_sorted, desc="Loading crops..."):
    labels = labels_dict[id]
    img = f"{img_source}/{id}.jpeg"
    img = cv2.imread(img)

    for idx, label in enumerate(labels):
      fname = store / f"{id}_{idx}.jpeg"
      imsize = img.shape[0]

      _, cx, cy, w, h, _= [int(i*imsize) for i in label]
      w, h = max(w,h), max(w, h)
      box = [cx-w//2, cy-w//2, cx+w//2, cy+w//2]
      box = [max(0, i) for i in box]
      crop = img[box[1]:box[3], box[0]:box[2]]

      crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
      cv2.imwrite(str(fname), crop)

def classify_imgstore(store, old_labels_dict, model_path):
  directory = Path(store)
  model = keras.models.load_model(model_path)
  files = sorted(list(directory.glob("*.jpeg")))
  images = np.zeros((len(files), 128, 128, 3), dtype=np.uint8)

  for index, file in tqdm(enumerate(files), "Loading images.."):
    file = str(file)
    image = cv2.imread(file)
    images[index] = image

  gen = DigitGen(images=images, out_size=128, batch_size=1024)
  labels = model.predict(gen, verbose=1)
  gen2 = DigitGen(images=255 - images, out_size=128, batch_size=1024)
  labels2 = model.predict(gen2, verbose=1)

  labels = labels + labels2
  labels = labels[:len(files)]
  print(labels.shape)

  fname_to_labels = {str(f.stem):l for f,l in zip(files, labels)}

  id_sorted = sorted(old_labels_dict.keys())
  missing = []
  for id in tqdm(id_sorted, desc="Loading crops..."):
    labels = old_labels_dict[id]
    for idx, label in enumerate(labels):
      # print(id, idx, label)
      fname = f"{id}_{idx}"
      if fname in fname_to_labels:
        y_ = fname_to_labels[fname]
        y_loc = np.argmax(y_)
        score = y_[y_loc]
        label[0] = y_loc
        label[-1] = score
      else:
        missing.append((fname, label))

  # print(len(missing))
  # print(missing[:10])
  # json.dump(missing, open("missing.json", 'w'), indent=3)

  return old_labels_dict

def labels_to_submission(label_dict, submission_file):
  header = "id,digit_sum\n"
  out = submission_file
  writer = open(out, 'w')
  writer.write(f"{header}")

  for id in tqdm(label_dict):
    pred_labels = label_dict[id]
    ans = sum(i[0] for i in pred_labels)
    sub_str = f"{id},{int(ans)}\n"
    writer.write(sub_str)
  writer.close()

def get_intersecting_labels(labels):
  bad_label_index = set()
  for i1, l1 in enumerate(labels):
    for i2, l2 in enumerate(labels):
      if i2 > i1:
        d1, x1, y1, w1, h1, conf1 = l1
        d2, x2, y2, w2, h2, conf2 = l2
        a1 = w1 * h1
        a2 = w2 * h2
        box1 = [x1 - w1/2, y1 - h1/2,  x1 + w1/2, y1 + h1/2]
        box2 = [x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2]

        iom = intersection_over_minArea(box1, box2)
        if iom > 0.999:
          if a1 < a2:
            bad_label_index.add(i1)
          if a2 < a1:
            bad_label_index.add(i2)

  return bad_label_index

def post_process(labels_dict):
  new_labels = {}
  for id in labels_dict:
    labels = labels_dict[id]
    new_labels[id] = []
    bad_indices = get_intersecting_labels(labels)
    for i3, l3 in enumerate(labels):
      if i3 not in bad_indices:
        new_labels[id].append(l3)

  new_labels = filter_by_confidence(new_labels)

  return new_labels

def filter_by_confidence(labels_dict):
  for id in labels_dict:
    labels = labels_dict[id]
    labels2 = []
    for label in labels:
      conf = label[-1]
      if conf > .9:
        labels2.append(label)
    labels_dict[id] = labels2
  return labels_dict

def measure(labels_dict, split):
  from yololib import draw_label_on_img
  cv2.namedWindow("img", flags=cv2.WINDOW_NORMAL)
  actual_labels = load_data(split)
  accuracy = 0
  total = 0

  id_sorted = sorted(labels_dict.keys())
  for id in tqdm(id_sorted, desc="Scoring..."):
    real_val = actual_labels[id]
    labels = labels_dict[id]
    # print(id, real_val, labels)
    # print(new_labels)
    pred_val = sum([p[0] for p in labels])
    if pred_val == real_val:
      accuracy += 1

    # else:
    #   impath = Path(f"data/{split}") / f"{id}.jpeg"
    #   img = cv2.imread(str(impath))
    #   cv2.imwrite(f"misclassified_train/{id}.jpeg", img)
    #   draw_label_on_img(img, labels)
    #   print(id, real_val)
    #   [print(l) for l in labels]
    #   print("--------------------------------")
    #   cv2.imshow("img", img)
    #   cv2.waitKey(-1)


    total += 1
  print(accuracy/total, total)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_labels", help="data dir", required=True)
    parser.add_argument("--split", help="train/fake_2560", required=True)
    args = parser.parse_args()

    from yololib import load_yolo_labels
    old_labels_dict = load_yolo_labels(args.yolo_labels)
    img_store = f"post_processing/digits_{args.split}/"
    labels_dict_json = f"labels_pred_{args.split}.json"

    # store_digit_images(labels_dict=old_labels_dict, split=args.split, store=img_store)
    labels_dict = classify_imgstore(store=img_store, old_labels_dict=old_labels_dict, model_path="small_mnist/")
    labels_dict = {i:np.asarray(l).tolist() for i,l in labels_dict.items()}
    json.dump(labels_dict, open(labels_dict_json, 'w'))
    labels_dict = json.load(open(labels_dict_json))
    labels_dict = post_process(labels_dict)
    labels_to_submission(labels_dict)
    measure(labels_dict, split=args.split)