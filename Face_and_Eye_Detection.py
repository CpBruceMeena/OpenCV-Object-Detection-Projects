"""
Face and Eye Detection
"""
import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier("C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/HaarCascade_XML files/haarcascade_frontalface_default.xml")
#Combining face and eye detection
eye_classifier = cv2.CascadeClassifier("C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/HaarCascade_XML files/haarcascade_eye.xml")

img = cv2.imread('C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/Required Images/sandeep.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 

"""
detectMultiScale(image, Scale Factor, Min Neighbors)
Scale Factor = Specifies how much we reduce the image size each time we scale. E.g. in face detection we typically use 1.3. This means we reduce the 
image by 30% each time it's scaled. Smaller values, like 1.05 will take longer to compute, but wil increase the rate of detection.

Min Neighbors: Specifies the numbero f neighors each potential window should have in order to consider it a positive detection.
Typically set between 3-6. It acts as sensitivity setting, low values will sometimes detect multiples faces over a single face.
High values willl ensure less false positives, but you may miss some faces.
"""

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

if faces is ():
    print("No faces found")
    
else:
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (127, 0, 255), 2)
        cv2.imshow("face detection", img)
        cv2.waitKey(0)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (255, 255, 0), 2)
            cv2.imshow('image', img)
            cv2.waitKey(0)       
#Combining face and eye detection
eye_classifier = cv2.CascadeClassifier("C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/HaarCascade_XML files/haarcascade_eye.xml")

cv2.destroyAllWindows()