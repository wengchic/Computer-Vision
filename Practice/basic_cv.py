import cv2
import matplotlib.pyplot as plt
import numpy as np

# cv2.waitKey(0)
# cv2.destroyAllWindows() 


img = cv2.imread('image/1.jpg')


##### 圖片預處理有很多種，包含模糊化(blur)、輪廓化(edge cascade)、膨脹、侵蝕
# blur = cv2.GaussianBlur(img, (5,5), 0)
# canny = cv2.Canny(blur, 125, 175)
# dilated = cv2.dilate(canny, (7,7), iterations=3)
# eroded = cv2.erode(canny, (3,3), iterations=1)


###### 移動圖片位置 Translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, transMat, dimensions)

# -x → 往左移動 ; x → 往右移動
# -y → 往上移動 ; y → 往下移動
#translated = translate(img, -100, 100)


###### 旋轉圖片 Rotate
def rotate(img, angle, rotPoint=None):
    (height,width) = img.shape[:2]
    
    if rotPoint is None:
        rotPoint = (width//2,height//2)
 
    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)
    
    return cv2.warpAffine(img, rotMat, dimensions)

#rotated = rotate(img, -45)


##### 圖片結構(輪廓與層級) Contours & Hierarchies
blank = np.zeros(img.shape, dtype='uint8')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
canny = cv2.Canny(blur, 125, 175)

contours, hierarchies = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 計算總共有幾個輪廓 contours
print(f'{len(contours)} contour(s) found!')

# 畫出當前所有的 contours
cv2.drawContours(img, contours, -1, (255,0,0), 1)
#cv2.imshow('Contours Drawn on img', img)

# 標示 contours
cv2.drawContours(blank, contours, -1, (0,255,0), 1)
#cv2.imshow('Contours Drawn on blank', blank)


# cv2.waitKey(0)
# cv2.destroyAllWindows() 

plt.imshow(img)
plt.show()
plt.imshow(blank)
plt.show()

















