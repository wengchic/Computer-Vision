import cv2
from matplotlib import pyplot as plt
import imutils


org_img = cv2.imread('image/2.jpg')
org_img = cv2.resize(org_img, (600, 480))
gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 12, 15, 15) #非線性的雙邊濾波器進行計算，讓影像模糊化的同時，也能夠保留影像內容的邊緣


###############方法一#######################
# # 用Canny尋找輪廓
# edged = cv2.Canny(gray, 30, 150)
# edged = cv2.dilate(edged, None)

# contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(contours)
# contours = sorted(contours, key = cv2.contourArea, reverse = True)[:20]
# screenCnt = None

# for c in contours:
#     # 如果approximated contour 找到4個角 代表可能就是車牌 
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.02 * peri, True)

#     if len(approx) == 4:
#         screenCnt = approx
#         break

# cv2.drawContours(img, [screenCnt], 0, (0, 255, 0), 2)
# cv2.imshow('Outline', img)
# cv2.waitKey(0)

###############方法二#######################
# 使用自定義的車牌辨識模型
plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# 檢測車牌位置
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

print(plates)

# 繪製車牌位置矩形
for (x, y, w, h) in plates:
    cv2.rectangle(org_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 顯示結果
cv2.imshow("License Plate Detection", org_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


