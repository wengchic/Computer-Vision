import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


def draw_plt(train_result, train_acc, val_acc):
	plt.plot(train_result.history[train_acc])
	plt.plot(train_result.history[val_acc])
	plt.title('train_acc vs val_acc')
	plt.ylabel('Train')
	plt.xlabel('Epoch')
	plt.legend(['train', 'validation'], loc='center right')
	plt.show()


# load data from mnist
# xtrain(60000, 28, 28) ytrain(60000,)
# xtest(10000, 28, 28) ytest (10000,)
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# train data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')																	# 轉換資料類型
x_train = x_train/255																				# Normalization 公式=(x - min)/(max - min) -> x / 255 (0~255 顏色範圍)           																							 

# test data
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test = x_test/255          



# CNN (Accuracy 0.9918)
CNN = keras.Sequential()
CNN.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
CNN.add(MaxPooling2D(2, 2)) 

CNN.add(Conv2D(64, (3, 3), activation = 'relu'))
CNN.add(MaxPooling2D(2, 2))

CNN.add(Dropout(0.25))

CNN.add(Flatten())

CNN.add(Dense(128, activation = 'relu'))
CNN.add(Dense(64, activation = 'relu'))
CNN.add(Dropout(0.5))
CNN.add(Dense(10, activation = 'softmax')) 															# 輸出0~9個不同種類數字

CNN.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

CNN.summary()
# keras.utils.plot_model(CNN, show_shapes = True)

# Training
train_result = CNN.fit(x_train, y_train, validation_data=(x_test, y_test), validation_split = 0.2, epochs = 5)
CNN.save('mnist_cnn.h5')

draw_plt(train_result, 'accuracy', 'val_accuracy')


score = CNN.evaluate(x_test, y_test)
print(score[1])
 