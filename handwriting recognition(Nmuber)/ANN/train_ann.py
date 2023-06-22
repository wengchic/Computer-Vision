import cv2
import numpy as np
import time
import tensorflow as tf


from tensorflow import keras
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# print(tf.test.is_gpu_available())
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# load data from mnist
# xtrain(60000, 28, 28) ytrain(60000,)
# xtest(10000, 28, 28) ytest (10000,)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

enc = OneHotEncoder(sparse_output=False, dtype=np.float32)

# train data
x_train = x_train.reshape(x_train.shape[0],-1)  # 轉換資料形狀 (60000, 784)
x_train = x_train.astype(np.float32)/255         # 轉換資料類型 Normalization 公式=(x - min)/(max - min) -> x / 255 (0~255 顏色範圍)   
#y_train = y_train.astype(np.float32)
y_train = enc.fit_transform(y_train.reshape(-1, 1))

# test data
x_test = x_test.reshape(x_test.shape[0],-1)    # (10000, 784)
x_test = x_test.astype(np.float32)/255          
#y_test = y_test.astype(np.float32)     
y_test = enc.fit_transform(y_test.reshape(-1, 1))


# ANN (Accuracy 0.915, Training Time：1568.773880 sec)
ANN = cv2.ml.ANN_MLP_create()
ANN.setLayerSizes(np.array([784, 512, 10]))
ANN.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ANN.setTermCriteria(( cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.001))
ANN.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
ANN.setBackpropWeightScale(0.0001)



# Training
print('Start Training...')
start = time.time()
ANN.train(x_train, cv2.ml.ROW_SAMPLE, y_train)  
ANN.save('mnist_ann.xml')
end = time.time()                       
print('Complete the training')

# Accuracy
print('Testing...')
_, y_hat_test = ANN.predict(x_test)

acc = accuracy_score(y_hat_test.round(), y_test)
                         
print('Accuracy', acc)
print("Training Time：%f sec" % (end - start))                        

