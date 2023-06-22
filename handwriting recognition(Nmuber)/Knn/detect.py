import cv2
import numpy as np

cap = cv2.VideoCapture(0)                  

knn = cv2.ml.KNearest_load('mnist_knn.xml')   
#ANN = cv2.ml.ANN_MLP_load('mnist_ann.xml')

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
    img = cv2.resize(img, (540, 300))                                           # 固定影像尺寸,加快處理速度       
    x, y, w, h = 400, 200, 80, 80                                               # 鎖定擷取數字的位置,大小
    img_num = img.copy()                                                        # 複製影像(辨識用)
    img_num = img_num[y:y+h, x:x+w]                                             # 擷取辨識的區域

    img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)                         # 轉灰階
    
    ret, img_num = cv2.threshold(img_num, 160, 255, cv2.THRESH_BINARY_INV)      # 二值化轉換 (黑白)
    output = cv2.cvtColor(img_num, cv2.COLOR_GRAY2BGR)                          # 輸出轉彩色
    img[10:90, 460:540] = output                                                # 輸出位置


    # Recognition
    img_num = cv2.resize(img_num, (28, 28))                                     # 轉成28x28 (和訓練模型一樣) 
    img_num = img_num.astype(np.float32)                                        # 轉換資料類型
    img_num = img_num.reshape(-1,)          
    img_num = img_num.reshape(1,-1)
    img_num = img_num/255
    

    img_pre = knn.predict(img_num)                                              # 辨識
    #img_pre = ANN.predict(img_num)


    output_num = str(int(img_pre[1][0][0]))                                     # 取得辨識結果

    text = output_num                              
    org = (x, y-20)                                                             # 文字位置
    font = cv2.FONT_HERSHEY_SIMPLEX                                             # 文字字體
    font_color = (20, 35, 255)                                                    # 文字顏色
    fontScale = 4                                                               # 文字大小

    border_thickness = 2                                                        # 邊框粗細
    border_Type = cv2.LINE_AA                                                   # 邊框樣式
    cv2.putText(img, text, org, font, fontScale, font_color, border_thickness, border_Type) 

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)                      # 標記辨識的區域
    cv2.imshow('TEST_knn', img)

    if cv2.waitKey(25) == ord('q'):                                             # q 鍵關閉
        break     

cap.release()
cv2.destroyAllWindows()