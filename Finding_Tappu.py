"""
Finding Waldo:
    It basically means that we are finding a template in a image
"""
import cv2
import numpy as np

image = cv2.imread('C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/Required Images/where_is_waldo.jpg')
cv2.imshow("Where is tappu", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

template = cv2.imread('C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/Required Images/waldo.jpg', 0)
cv2.imshow("waldo", template)
cv2.waitKey(0)

result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#Creating Bounding Box
top_left = max_loc
bottom_right = (top_left[0]+50, top_left[1]+50)
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 5)

cv2.imshow("waldo found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()