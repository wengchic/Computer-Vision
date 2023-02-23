import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('Stereo_images/image21.png')
img2 = cv2.imread('Stereo_images/image22.png')

gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)


'''
FLANN的匹配(FLANN based Matcher)
1.FLANN代表近似最近鄰居的快速庫。代表一組經過優化的算法，用於大數據集中的快速最近鄰搜索以及高維特徵。
2.對於大型數據集，它的工作速度比BFMatcher快。
3.需要傳遞兩個字典來指定要使用的算法及其相關參數等
對於SIFT或SURF等算法，可以用以下方法：
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

對於ORB，可以使用以下參數：
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12   這個參數是searchParam,指定了索引中的樹應該遞歸遍歷的次數。值越高精度越高
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
'''

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50) # It specifies the number of times the trees in the index should be recursively traversed

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k = 2)  


pts1 = []
pts2 = []

# Threshold 0.8 (ratio test as per Lowe's paper)
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# select inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

def drawlines(img1, img2, lines, pts1, pts2):
    r,c,l = img1.shape # height, width, channels, if img is gray：height, width
    

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]]) #y = -(a/b)x - c/b, if x = 0 then y = -c/b
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        print('x0,y0', x0, y0)
        print('x1,y1', x1, y1)
        img1 = cv2.line(img1, (x0,y0),(x1,y1), color, 2)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, 1)

    return img1, img2

# Find epilines corresponding to points in right image (img2)
# drawing its lines on left image
# computeCorrespondEpilines: Output vector of the epipolar lines corresponding to the points in the other image. 
# Each line ax+by+c=0 is encoded by 3 numbers (a,b,c) .
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 1, F)
lines1 = lines1.reshape(-1,3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (img1)
# drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
# lines2 = lines2.reshape(-1,3)
# img3, img4 = drawlines(img1, img2, lines2, pts1, pts2)


# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()


