import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #lower_blue = np.array([38, 86, 0])
    #upper_blue = np.array([121, 255, 255])
    #mask = cv2.inRange(hsv, lower_blue, upper_blue)

    #contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows() 