import cv2
import numpy as np
import time

from tensorflow import keras
from keras.datasets import mnist



# load data from mnist
# xtrain(60000, 28, 28) ytrain(60000,)
# xtest(10000, 28, 28) ytest (10000,)
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# train data
x_train = x_train.reshape(x_train.shape[0],-1)  # 轉換資料形狀 (60000, 784)
x_train = x_train.astype('float32')/255         # 轉換資料類型 Normalization 公式=(x - min)/(max - min) -> x / 255 (0~255 顏色範圍)   
y_train = y_train.astype(np.float32)

# test data
x_test = x_test.reshape(x_test.shape[0],-1)    # (10000, 784)
x_test = x_test.astype('float32')/255          
y_test = y_test.astype(np.float32)



# Knn (Accuracy 0.9688, Training Time：4.703460 sec)
knn = cv2.ml.KNearest_create()                  
knn.setDefaultK(5)                              
knn.setIsClassifier(True)


# Training
print('Start Training...')
start = time.time()
knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)  
knn.save('mnist_knn.xml')
end = time.time()                       
print('Complete the training')

# Accuracy
print('Testing...')
test_pre = knn.predict(x_test)                  
test_ret = test_pre[1]
test_ret = test_ret.reshape(-1,)
test_sum = (test_ret == y_test)

acc = test_sum.mean()                           

print('Accuracy', acc)
print("Training Time：%f sec" % (end - start))