"""
MINI-PROJECT = Counting Circles and Ellipses
cv2.SimpleBlobDetector_Params()
Area:
    params.filterByArea = True/False
    params.minArea = pixels
    params.maxArea = pixels
    
Circularity:
    params.filterByCircularity = True/False
    params.minCircularity = 1 being perfect circle, 0 the opposite
    
Convexity - Area of blob/ Area of Convex Hull:
    params.filterBuConvexity = True/False
    params.minConvexity = 0 to 1
    
Intertia - Measure of ellipticalness (low being more elliptical, high being more circular):
    params.filterByInertia = True/False
    params.minInertiaRatio = 0.01
"""

import cv2
import numpy as np

#to load an image 
image = cv2.imread('C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/Required Images/circles_ellipses.png', 0)
cv2.imshow("image", image)
cv2.waitKey(0)

#Set up the detector with default parameters
detector = cv2.SimpleBlobDetector_create()

#Detect blobs
keypoints = detector.detect(image)

#Draw detected blobs as red circles,
#cv2.DRAW_MATCHES_FLAGS_
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Total no of blobs:" + str(len(keypoints))

cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

cv2.imshow("Blobs using default parameters", blobs)

#It will show the window until the user press a key
cv2.waitKey(0)

params = cv2.SimpleBlobDetector_Params()

#Set Area filtering parameters()
params.filterByArea = True
params.minArea = 50

#Set Circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.89

#Set convexity filtering parameters
params.filterByConvexity = False
params.minConvexity = 0.1

#Set Inertial filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.01

#Create a detector with parameters
detector = cv2.SimpleBlobDetector_create(params)

#Create blobs
keypoints = detector.detect(image)

#Draw blobs on our image as red circles
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Total no of blobs:" + str(len(keypoints))

cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

cv2.imshow("Filtering circular blobs", blobs)

cv2.waitKey(0)
#when you have seen all the windows press ESC at the last to destroy all the windows
#the waitKey(0) means it is waiting for ESC key
#waitKey(1) == 13 means press ENTER
#This closes all windows
cv2.destroyAllWindows()

