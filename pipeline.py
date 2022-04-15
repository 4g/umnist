from geomlib import CachedLocationGenerator
import make_yolo_data
from yolov5.detect import run as yolo_detect_run
from yolov5.train import run as yolo_train_run
import create_classifier_data
from pathlib import Path
from train_mnist import train as train_classifier
from settings import CFG, Paths
from yololib import load_yolo_labels
import measurelib as reclassify
import numpy as np
import shutil

def make_dirs():
  for dir in Paths.dirs:
    p = Path(dir)
    p.mkdir(parents=True, exist_ok=True)

def pipeline(task):
  if task == "train_yolo":
    locgen = CachedLocationGenerator(size=CFG.locgen_img_size,
                                     ratio=20,
                                     minsize=0.03,
                                     maxsize=0.5,
                                     iters=CFG.locgen_samples)
    locgen.create()
    locgen.save(Paths.locgen)

    # Create synthetic data of images with digits and cropped digits
    # Full image is used for training yolo small which predicts locations and classes.
    # Using yolo small due to large image size,
    # But once the location of digits has been detected, we can use a large classifier to classify them
    # To train this classifier we crop the digits from our synthetic dataset

    make_yolo_data.create_and_save(img_width=CFG.sythetic_img_size,
                       impath = f"{Paths.generated_data}/images/train/",
                       labels_path=f"{Paths.generated_data}/labels/train/",
                       n_samples=CFG.synthetic_img_samples,
                       prefix="0")

    make_yolo_data.create_and_save(img_width=CFG.sythetic_img_size,
                       impath=f"{Paths.generated_data}/images/train/",
                       labels_path=f"{Paths.generated_data}/labels/train/",
                       n_samples=CFG.synthetic_empty_image_samples,
                       prefix='empty')


    make_yolo_data.create_and_save(img_width=CFG.sythetic_img_size,
                       impath = f"{Paths.generated_data}/images/val/",
                       labels_path=f"{Paths.generated_data}/labels/val/",
                       n_samples=CFG.synthetic_val_img_samples,
                       prefix="0")

    make_yolo_data.make_yaml(base_path=f"{Paths.generated_data}",
                             yolo_cfg_path=Paths.yolo_cfg)

    create_classifier_data.create(img_width=CFG.sythetic_img_size,
                                  n_samples=CFG.digit_samples,
                                  net_img_size=CFG.digit_size,
                                  location=Paths.generated_digits)

    yolo_train_run(data=Paths.yolo_cfg,
                   imgsz=CFG.sythetic_img_size,
                   weights='yolov5s.pt',
                   batch_size=-1,
                   hyp=Paths.yolo_hyp,
                   project=Paths.yolo_model,
                   name="detector",
                   exist_ok=True,
                   epochs=CFG.n_yolo_epochs)


  if task == "train_classifier":
    train_classifier(model_path=Paths.classifier_model, img_size=CFG.digit_size, num_epochs=CFG.n_classifier_epochs)


  if task == "infer" or task == "infer_pretrained":
    if task == "infer_pretrained":
      Paths.yolo_model = Paths.pretrained_yolo
      Paths.classifier_model = Paths.pretrained_classifier

    yolo_detect_run(weights=f"{Paths.yolo_model}/detector/weights/best.pt",
               source=f"{Paths.comp_data}/test/",
               imgsz=(CFG.sythetic_img_size, CFG.sythetic_img_size),
               conf_thres=0.1,
               iou_thres=0.1,
               max_det=5,
               agnostic_nms=True,
               save_txt=True,
               save_conf=True,
               nosave=True,
               project=Paths.comp_data,
               name="infer",
               exist_ok=True,
               half=True)

    yolo_infer_labels = f"{Paths.comp_data}/infer/labels/"
    old_labels_dict = load_yolo_labels(yolo_infer_labels)
    # # print(old_labels_dict)
    img_store = f"{Paths.post_processing}/digits/"
    # # # labels_dict_json = f"labels_pred_{args.split}.json"
    # #
    reclassify.store_digit_images(labels_dict=old_labels_dict, store=img_store, img_source=f"{Paths.comp_data}/test/")

    labels_dict = reclassify.classify_imgstore(store=img_store, old_labels_dict=old_labels_dict, model_path=Paths.classifier_model)
    labels_dict = {i: np.asarray(l).tolist() for i, l in labels_dict.items()}

    # import json
    # json.dump(labels_dict, open("temp.json", 'w'))
    # labels_dict = json.load(open("temp.json"))

    labels_dict = reclassify.post_process(labels_dict)
    reclassify.labels_to_submission(labels_dict, Paths.submission_file)
    # reclassify.measure(labels_dict, split=args.split)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", help="train_yolo/train_classifier/set_pretrained/infer/infer_pretrained")

  args = parser.parse_args()
  pipeline(args.task)









