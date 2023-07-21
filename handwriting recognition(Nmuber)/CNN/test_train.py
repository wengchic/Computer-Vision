import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation
from tensorflow_addons.optimizers import AdamW


def draw_plt(train_result, train_acc, val_acc):
    plt.plot(train_result.history[train_acc])
    plt.plot(train_result.history[val_acc])
    plt.title('train_acc vs val_acc')
    plt.ylabel('Train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.show()

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 20, 30 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3

    if epoch >= 30:
        lr *= 1e-2
    elif epoch >= 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def wd_schedule(epoch):
    """Weight Decay Schedule
    Weight decay is scheduled to be reduced after 20, 30 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        wd (float32): weight decay
    """
    wd = 1e-4

    if epoch >= 30:
        wd *= 1e-2
    elif epoch >= 20:
        wd *= 1e-1
    print('Weight decay: ', wd)
    return wd


# load data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# train data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_train = x_train / 255.0

# test data
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test = x_test / 255.0

optimizer = AdamW(learning_rate=lr_schedule(0), weight_decay=wd_schedule(0))

# CNN (Accuracy 0.993)
CNN = Sequential()

CNN.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
CNN.add(BatchNormalization())
CNN.add(Activation('swish'))
CNN.add(MaxPooling2D(2, 2))

CNN.add(Conv2D(64, (3, 3)))
CNN.add(BatchNormalization())
CNN.add(Activation('swish'))
CNN.add(MaxPooling2D(2, 2))

CNN.add(Dropout(0.25))

CNN.add(Flatten())

CNN.add(Dense(128))
CNN.add(BatchNormalization())
CNN.add(Activation('swish'))

CNN.add(Dense(64))
CNN.add(BatchNormalization())
CNN.add(Activation('swish'))

CNN.add(Dropout(0.5))
CNN.add(Dense(10, activation='softmax'))

CNN.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

CNN.summary()

# Training
train_result = CNN.fit(x_train, y_train, validation_data=(x_test, y_test), validation_split=0.2, epochs=5)
CNN.save('mnist_cnn.h5')

draw_plt(train_result, 'accuracy', 'val_accuracy')

score = CNN.evaluate(x_test, y_test)
print("Test accuracy:", score[1])