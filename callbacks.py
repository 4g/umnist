from tensorflow import keras
from datetime import datetime


def lr_schedule():
    def lrs(epoch):
        if epoch < 50:
            return 0.0003
        if epoch < 100:
            return 0.0001
        if epoch < 150:
            return 0.00003
        else:
          return 0.00001

    return keras.callbacks.LearningRateScheduler(lrs, verbose=True)

def checkpoint(filepath):
    return keras.callbacks.ModelCheckpoint(filepath=filepath,
                                        monitor='val_loss',
                                        save_weights_only=True,
                                        save_best_only=False,
                                        verbose=1)



def tensorboard():
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, profile_batch='10,20')
    return tensorboard_callback
