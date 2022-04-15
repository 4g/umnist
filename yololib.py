from pathlib import Path
import cv2

def load_yolo_labels(root_dir):
  labels = Path(root_dir).glob("*.txt")
  label_dict = {}
  for label in labels:
    label_dict[label.stem] = [[float(i) for i in x.strip().split()] for x in open(label)]
  return label_dict

def draw_label_on_img(img, labels):
  imsize = img.shape[0]
  for pred in labels:
    if len(pred) > 5:
      pred = pred[:5]
    label, cx, cy, w, h = pred
    _, cx, cy, w, h = [int(i * imsize) for i in pred]
    box = [cx - w // 2, cy - w // 2, cx + w // 2, cy + w // 2]
    cv2.rectangle(img, pt1=(box[0], box[1]), pt2=(box[2], box[3]), thickness=4, color=(0, 255, 0))
    cv2.putText(img, org=(cx, cy), text=str(int(label)), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN,
                color=(0, 0, 255), thickness=3)


def render(labels_dir, image_dir):
  label_dict = load_yolo_labels(labels_dir)

  cv2.namedWindow("image", flags=cv2.WINDOW_NORMAL)
  image_dir = Path(image_dir)
  for id in label_dict:
    impath = image_dir / f"{id}.jpeg"
    img = cv2.imread(str(impath))
    # img = cv2.resize(img, dsize=None, fx=0.25, fy=0.25)
    draw_label_on_img(img, label_dict[id])
    cv2.imshow("image", img)
    cv2.waitKey(-1)


def intersection_over_minArea(boxA, boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  # compute the area of intersection rectangle
  interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
  if interArea == 0:
    return 0
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
  boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(min(boxBArea, boxAArea))

  # return the intersection over union value
  return iou


# def labels_to_sum_csv(labels_dir, csv_path):
#   labels = load_yolo_labels(labels_dir)
#   for id in labels:
