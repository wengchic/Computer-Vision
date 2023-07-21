import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def preprocess_image(img):
    img = cv2.resize(img, (540, 300))                                           # 固定畫面尺寸,加快處理速度       
    x, y, w, h = 400, 200, 80, 80                                               # 鎖定擷取數字的位置,大小
    img_num = img.copy()                                                        # 複製影像(辨識用)
    img_num = img_num[y:y+h, x:x+w]                                             # 擷取辨識的區域

    img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)                         # 轉灰階
    img_num = cv2.adaptiveThreshold(img_num, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)      # 二值化轉換 (黑白)
    output = cv2.cvtColor(img_num, cv2.COLOR_GRAY2BGR)                          # 輸出轉彩色
    img[10:90, 460:540] = output                                                # 輸出位置

    return img, img_num, (x, y, w, h)

def recognition_image(img_num):
    img_num = cv2.resize(img_num, (28, 28))                                     # 轉成28x28 (和訓練模型一樣) 
    img_num = img_num/255
    img_num = img_num.reshape(1, 28, 28, 1)                                               
    img_pred = CNN.predict(img_num)                                             # 辨識
    img_pred = np.argmax(img_pred, axis = 1)
    output_num = str(int(img_pred))                                             # 取得辨識結果

    return output_num

def preprocess_text(img, output_num, x, y, w, h):
    text = output_num                             
    org = (x, y-20)                                                             # 文字位置
    font = cv2.FONT_HERSHEY_SIMPLEX                                             # 文字字體
    font_color = (20, 35, 255)                                                  # 文字顏色
    fontScale = 4                                                               # 文字大小
    border_thickness = 2                                                        # 邊框粗細
    border_Type = cv2.LINE_AA                                                   # 邊框樣式
    
    cv2.putText(img, text, org, font, fontScale, font_color, border_thickness, border_Type) 
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)                      # 標記辨識的區域
    cv2.imshow('TEST_CNN', img)



cap = cv2.VideoCapture(0)                  
CNN = load_model('mnist_cnn.h5', compile = True)

print('Start......')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, img = cap.read()

    if not ret:
        print("Unable to receive frame")
        break


    # Image processing
    img, img_num, (x, y, w, h) = preprocess_image(img)

    if img_num is not None:
        # Recognition
        output_num = recognition_image(img_num)

        # Text
        preprocess_text(img, output_num, x, y, w, h)

    if cv2.waitKey(25) == ord('q'):                                               # q 鍵關閉                                             
        break     


cap.release()
cv2.destroyAllWindows()