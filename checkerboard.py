import numpy as np
import cv2
import albumentations as A
import random

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def make_checkerboard(im_width=512):
    size = random.randint(10, 10)
    x = np.random.rand(size, size)
    x[x >= 0.5] = 255
    x[x < 0.5] = 0
    x = x.astype(np.uint8)
    x = cv2.resize(x, dsize=(im_width, im_width), interpolation=cv2.INTER_NEAREST)

    transform = A.Compose([
        A.Rotate(limit=20, p=0.75, interpolation=cv2.INTER_CUBIC),
        A.Perspective(scale=(0.3, 0.4), interpolation=cv2.INTER_CUBIC, p=0.75),
    ])

    x = transform(image=x)['image']
    # angle = random.randint(0, 45)
    # x = rotate_image(x, angle)
    x = cv2.resize(x, dsize=(im_width, im_width), interpolation=cv2.INTER_CUBIC)

    radii_ratios = (0.1, 0.3)
    n_circles = random.randint(3, 5)
    radius_small = int(im_width*radii_ratios[0])
    radius_large = int(im_width*radii_ratios[1])

    tmp_img = np.zeros((im_width, im_width), dtype=np.uint8)

    for i in range(n_circles):
        radius = random.randint(radius_small, radius_large)
        center_x = random.randint(0, im_width)
        center_y = random.randint(0, im_width)
        color = 255 if random.random() > 0.5 else 0
        cv2.circle(img=tmp_img, center=(center_x, center_y), radius=radius, thickness=-1, color=color)

    x = cv2.bitwise_xor(x, tmp_img)

    n_triangles = random.randint(3, 5)
    tmp_img = np.zeros((im_width, im_width), dtype=np.uint8)
    rpoint = lambda: (random.randint(0, im_width), random.randint(0, im_width))
    for i in range(n_triangles):
        n_points = random.randint(3, 5)
        points = [rpoint() for i in range(n_points)]
        points = np.array(points)
        color = 255 if random.random() > 0.5 else 0
        cv2.fillPoly(tmp_img, pts=[points], color=color)

    x = cv2.bitwise_xor(x, tmp_img)
    return x

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    from tqdm import tqdm

    cv2.namedWindow("img", flags=cv2.WINDOW_NORMAL)
    for i in tqdm(range(3000)):
        x = make_checkerboard(4000)
        cv2.imshow("img", x)
        cv2.waitKey(-1)