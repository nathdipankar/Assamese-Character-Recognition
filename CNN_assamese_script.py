# -*- coding: utf-8 -*-
"""
Created on Thu May 17 23:28:52 2018

@author: nathd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import metrics
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau


model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Dropout(0.25))

# the model so far outputs 3D feature maps (height, width, features)

# dense layer
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(736, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(183, activation = 'softmax'))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

batch_size = 30

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(50, 50),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(50, 50),
        batch_size=batch_size,
        class_mode='categorical')

model.summary()

history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=200,
        validation_data=validation_generator,
        validation_steps=1000 // batch_size)
model.save_weights('first_try.h5')

fig, ax = plt.subplots(1,1, figsize = (12,9))
ax.plot(history.history['loss'], '-', color = 'xkcd:magenta', label = 'Loss', alpha = 0.8)
ax.plot(history. history['val_loss'], '--', color = 'xkcd:black', label = 'Validation Loss', alpha = 0.8)
ax.legend(loc = 'best')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_facecolor('0.9')

from resizeimage import resizeimage
def predict_new_image(path):
        test1 = load_img(path)
        test2 = resizeimage.resize_contain(test1, [50,50]).convert('RGB')
        test = img_to_array(test2)
        test = np.expand_dims(test, axis = 0)
        result = model.predict(test)
        x = train_generator.class_indices
        all_label = list(x.keys())
        plt.figure()
        plt.clf()
        plt.imshow(test2, cmap = 'Greys')
        plt.title(all_label[result.argmax()], fontsize = 15)
predict_new_image('data/predict/predict_1.jpg')
predict_new_image('data/predict/predict_2.jpg')
predict_new_image('data/predict/predict_3.jpg')
predict_new_image('data/predict/predict_4.jpg')
predict_new_image('data/predict/predict_7.jpg')
predict_new_image('data/predict/predict_5.jpg')
predict_new_image('data/predict/predict_6.jpg')
predict_new_image('data/predict/predict_8.jpg')
predict_new_image('data/predict/predict_9.jpg')
predict_new_image('data/predict/predict_11.jpg')

predict_generator = test_datagen.flow_from_directory(
                'data/test',
                target_size=(50, 50),
                batch_size=1649,
                class_mode='categorical', shuffle = False)
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#Predict the output
y_pred = model.predict_generator(predict_generator)
#y_pred_classes
y_pred_classes = np.argmax(y_pred, axis = 1)

#y true values

y_true = validation_generator.classes

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize = (20, 20))
plot_confusion_matrix(cm, classes = range(183)) 