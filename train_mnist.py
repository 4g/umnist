import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import albumentations as A
import cv2
import random
from keras.applications import efficientnet_v2
import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

class MnistGen(keras.utils.Sequence):
  def __init__(self, images, labels, batch_size, out_size, augment=True, shuffle=True):
    self.images = images
    self.labels = labels
    self.data = list(range(len(self.images)))
    self.augment = augment
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.out_size = out_size
    self.imgb = np.zeros((self.batch_size, self.out_size, self.out_size, 3), dtype=np.float32)
    self.maskb = np.zeros((self.batch_size, 10), dtype=np.float32)
    self.transform = A.Compose([
      A.RandomBrightnessContrast(p=0.5),
      A.HueSaturationValue(),
      A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_AREA),
            A.Perspective(p=0.2, interpolation=cv2.INTER_AREA),
            A.InvertImg(p=0.5),
            A.Blur(p=0.25),
        ], p=0.5)

  def __len__(self):
    return len(self.data) // self.batch_size

  def __getitem__(self, item):
    batch = self.data[item*self.batch_size:(item + 1)*self.batch_size]
    for idx, dindex in enumerate(batch):
      image = self.images[dindex]
      count = self.labels[dindex]
      if self.augment:
        image = self.transform(image=image)["image"]
      image = cv2.resize(image, dsize=(self.out_size, self.out_size))
      self.imgb[idx,:,:] = image.astype(np.float32) / 255
      self.maskb[idx] = count

    return self.imgb, self.maskb

  def on_epoch_end(self):
    if self.shuffle:
      random.shuffle(self.data)


def lr_schedule():
    def lrs(epoch):
        if epoch < 20:
            return 0.00003
        if epoch < 40:
            return 0.00003
        else:
            return 0.00003

    return keras.callbacks.LearningRateScheduler(lrs, verbose=True)

def train(model_path, img_size, num_epochs):
  # Model / data parameters
  num_classes = 10

  # the data, split between train and test sets
  from create_classifier_data import load_dataset
  (x_train, y_train), (x_test, y_test) = load_dataset()

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)


  model = efficientnet_v2.EfficientNetV2B1(input_shape=(img_size, img_size, 3),
                                        include_top=False,
                                        include_preprocessing=False)

  x = layers.GlobalAveragePooling2D(keepdims=True)(model.output)
  x = layers.Conv2D(
    1280,
    kernel_size=1,
    padding='same',
    use_bias=True,
    name='Conv_2')(x)
  x = layers.Activation(activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Conv2D(num_classes, kernel_size=1, padding='same', name='Logits')(x)
  x = layers.Flatten()(x)
  x = layers.Activation(activation='softmax',
                        name='Predictions')(x)

  model = keras.models.Model(inputs=model.input, outputs=x)

  # model.summary()

  batch_size = 64

  # train_gen = MnistGen(images=x_train, labels=y_train, batch_size=batch_size, out_size=img_size, augment=True, shuffle=True)
  val_gen = MnistGen(images=x_test, labels=y_test, batch_size=256, out_size=img_size, augment=False, shuffle=False)

  epochs = num_epochs

  model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(.0003), metrics=["accuracy"])

  # model = keras.models.load_model("small_mnist/")
  from callbacks import lr_schedule

  # comment this before final train
  # model.fit(x=train_gen, validation_data=val_gen, epochs=epochs, callbacks=[lr_schedule()])

  x_train = np.concatenate((x_train, x_test))
  y_train = np.concatenate((y_train, y_test))

  train_gen = MnistGen(images=x_train, labels=y_train, batch_size=batch_size, out_size=img_size, augment=True, shuffle=True)

  model.fit(x=train_gen, validation_data=val_gen, epochs=epochs, callbacks=[lr_schedule()])

  model.save(model_path)

  model = keras.models.load_model(model_path)

  x_test = [cv2.resize(i, (img_size, img_size)) for i in x_test]
  x_test = np.asarray(x_test, dtype=np.float32) / 255.
  score = model.evaluate(x_test, y_test, verbose=0)
  print("Test loss:", score[0])
  print("Test accuracy:", score[1])
