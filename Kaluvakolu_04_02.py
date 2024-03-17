import pytest
import numpy as np
from Kaluvakolu_04_01 import CNN
import os
import os
import tensorflow as tf
import numpy as np
import keras
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import load_model
from keras.optimizers import SGD, RMSprop, Adagrad
import tensorflow as tf
import numpy as np
import keras
# import tensorflow.keras as keras
from keras.callbacks import History
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD ,RMSprop ,Adagrad
from keras.utils import to_categorical


def test_evaluate():
    cnn = CNN()
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    samples = 100

    X_train = X_train[0:samples, :]
    X_train = X_train.astype('float32') / 255

    test_samples = 10
    X_test = X_test[0:test_samples, :]
    X_test = X_test.astype('float32') / 255



    y_train = to_categorical(y_train, 10)
    y_train = y_train[0:samples, :]
    y_test = to_categorical(y_test, 10)
    y_test = y_test[0:test_samples, :]
    cnn.add_input_layer(shape=(32, 32, 3))
    cnn.append_conv2d_layer(num_of_filters=64, kernel_size=(
        3, 3), activation='relu', name="conv01")
    cnn.append_conv2d_layer(num_of_filters=32, kernel_size=(
        3, 3), activation='relu', name="conv02")
    cnn.append_flatten_layer(name="flat01")
    cnn.append_dense_layer(num_nodes=10, activation="relu", name="dense1")
    cnn.set_optimizer(optimizer="SGD")
    cnn.set_loss_function(loss="hinge")
    cnn.set_metric(metric='accuracy')

    cnn.train(X_train=X_train, y_train=y_train,batch_size=None, num_epochs=2)
    (loss, ini_metric) = cnn.evaluate(X=X_test, y=y_test)
    cnn.save_model(model_file_name="model")
    cnn.load_a_model(model_file_name="model")
    cnn.train(X_train=X_train, y_train=y_train,batch_size=None, num_epochs=2)
    (loss, second_metric) = cnn.evaluate(X=X_test, y=y_test)
    assert ini_metric<=second_metric

from keras.datasets import cifar10, cifar100
def test_train():
    cnn = CNN()
    batch_size=32
    num_epochs=5
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    samples = 320

    X_train = X_train[0:samples, :]
    X_train = X_train.astype('float32') / 255

    test_samples = 32
    X_test = X_test[0:test_samples, :]
    X_test = X_test.astype('float32') / 255

    y_train = to_categorical(y_train, 10)
    y_train = y_train[0:samples]
    y_test = to_categorical(y_test, 10)
    y_test = y_test[0:test_samples]
    cnn.add_input_layer(shape=(32, 32, 3))
    cnn.append_conv2d_layer(num_of_filters=64, kernel_size=(3, 3), activation='relu', name="conv1")
    cnn.append_conv2d_layer(num_of_filters=32, kernel_size=(3, 3), activation='relu', name="conv2")
    cnn.append_flatten_layer(name="flat01")
    cnn.append_dense_layer(num_nodes=10, activation="relu", name="dense1")
    cnn.set_optimizer(optimizer="SGD")
    cnn.set_loss_function(loss="hinge")
    cnn.set_metric(metric='accuracy')
    LossList = cnn.train(X_train=X_train, y_train=y_train, batch_size=batch_size, num_epochs=num_epochs)
    assert LossList is not None
    for i in range(num_epochs-1):
        assert LossList[i]>LossList[i+1]