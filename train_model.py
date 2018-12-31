from sklearn.model_selection import train_test_split
import numpy as np
import keras
import os
from alexnet import alexnet
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import random
import cv2
from aug_funcs import zoom, pan, img_random_brightness

# load our image and label paths. Set the batch size #
image_paths = np.load('data//image_paths.npy')
label_paths = np.load('data//label_paths.npy')
batch = 100

### split our paths into train/valid sets ###
X_train, X_valid, y_train, y_valid = train_test_split(
    image_paths, label_paths, test_size=0.2, random_state=6)

### data augmenter ###
### feel free to add more augmenters ###


def random_augment(screen):

    if np.random.rand() < 0.5:
        screen = pan(screen)
    if np.random.rand() < 0.5:
        screen = zoom(screen)
    if np.random.rand() < 0.5:
        screen = img_random_brightness(screen)

    return screen

# since data might get too big to fit into RAM at some point(depending how much you collected)
# we will use our custom made batch_generator to feed our data to gpu in batches
# we will pass 100 image paths and labels to model for now, open them, augment them, train on them
# and then repeat the cycle


def batch_generator(image_paths, label_paths, batch_size, is_training):

    while True:
        batch_screen = []
        batch_label = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            if is_training:
                screen = random_augment(np.load(image_paths[random_index]))
            else:
                screen = np.load(image_paths[random_index])
            label = np.load(label_paths[random_index])
            screen = cv2.GaussianBlur(screen, (3, 3), 0)
            screen = screen.reshape(224, 224, 1)

            batch_screen.append(screen)
            batch_label.append(label)

        yield (np.asarray(batch_screen), np.asarray(batch_label))


### load the model ###
### using alexnet here, but you can specify your own model ###
model = alexnet()

# steps_per_epoch: Total number of steps (batches of samples)
# to yield from generator before declaring one epoch finished and starting the next epoch.
# It should typically be equal to the number of unique samples of your dataset divided by the batch size.
# Since we're augmenting our data we can specify even bigger number of steps
# In my case I multiplied number of steps by value of 10, you can start with lower number. Validation should be somewhat lower.
steps_per_epoch = int(np.ceil(X_train.shape[0] / batch))


# We will augment only our train data(mind the 1(True) for train and 0(False) for validation)
# Feel free to experiment with parameters!

hst = model.fit_generator(batch_generator(X_train, y_train, batch, 1),
                          steps_per_epoch=steps_per_epoch*10,
                          epochs=15,
                          validation_data=batch_generator(
    X_valid, y_valid, batch, 0),
    validation_steps=steps_per_epoch*7,


    verbose=1,
    shuffle=1)

model.save('model_new_2.h5')


### matplotlib for visualization ###
plt.plot(hst.history['loss'])
plt.plot(hst.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.show()
