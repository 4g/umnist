class Paths:
  base = "store"
  comp_data = f"{base}/data/competition/"
  yolo = f"{base}/data/yolo/"
  yolo_cfg = f"cfgs/umnist.yaml"
  yolo_hyp = f"cfgs/hyp.yaml"
  generated_data = f"{base}/data/generated/"
  generated_digits = f"{base}/data/digits/"
  post_processing = f"{base}/post_processing/"
  locgen = f"{base}/locgen.npy"
  yolo_model = f"{base}/models/yolo/"
  classifier_model = f"{base}/models/classifier/"
  submission_file = f"yolo_submission.csv"

  pretrained_yolo = "pretrained/"
  pretrained_classifier = "pretrained/small_mnist/"

  dirs = [base, comp_data, yolo, generated_data, yolo_model, classifier_model]

class CFG:
  locgen_img_size = 1024
  locgen_samples = 30000
  sythetic_img_size = 2560
  synthetic_img_samples = 30000
  synthetic_val_img_samples = 2000
  synthetic_empty_image_samples = 4000
  digit_samples = 40000
  digit_size = 128
  n_classifier_epochs = 200
  n_yolo_epochs = 100

#
# class CFG:
#   locgen_img_size = 1024
#   locgen_samples = 100
#   sythetic_img_size = 2560
#   synthetic_img_samples = 400
#   synthetic_val_img_samples = 20
#   synthetic_empty_image_samples = 4
#   digit_samples = 40
#   digit_size = 128
#   n_classifier_epochs = 1
#   n_yolo_epochs = 1
