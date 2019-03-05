"""
Car Detection
Try to use a low quality and low size video
"""

import numpy as np
import cv2

#Create a body classifier 
car_classifier = cv2.CascadeClassifier("C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/HaarCascade_XML files/haarcascade_car.xml")

#Initiate vidoe capture for video file
#Find an appropriate video for car detection
cap = cv2.VideoCapture("C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/Required Images/car.webm")
 
#Loop once video is openes
while cap.isOpened():
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cars = car_classifier.detectMultiScale(gray, 1.3, 3)
    
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow("cars ", frame)
        
    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()
