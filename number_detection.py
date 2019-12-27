from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from utils import crop_img


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class NumberDetection():
    def train(self):
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # Add a channels dimension
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(10000).batch(32)

        test_ds = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(32)
        # Create an instance of the model
        model = MyModel()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5, verbose=2)
        model.evaluate(x_test, y_test, verbose=2)
        model.save("number_detection")

    def predict(self, img):
        model = load_model("number_detection")
        pred_img = np.expand_dims(cv2.resize(
            img, (28, 28)), axis=2) / 255.0
        pred = model.predict(np.expand_dims(pred_img, axis=0))
        return np.argmax(pred)


class OptionDetection():
    def __init__(self, img, option_boxes):
        self.img = img
        self.option_boxes = option_boxes

    def evaluate(self):
        boxes = [crop_img(self.img, option_box, border=18)
                 for option_box in self.option_boxes]
        option = np.argmax([box.mean() for box in boxes])
        option_number = NumberDetection().predict(boxes[option])

        exam_number = {}
        exam_number["subject"] = "I" if option == 0 else "F"
        exam_number["subject_number"] = option_number

        return exam_number
