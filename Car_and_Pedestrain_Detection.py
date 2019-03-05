"""
Pedestrain Detection
Try to use a low quality and low size video
"""

import numpy as np
import cv2

#Create a body classifier 
body_classifier = cv2.CascadeClassifier("C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/HaarCascade_XML files/haarcascade_fullbody.xml")

#Initiate vidoe capture for video file
cap = cv2.VideoCapture("C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/Required Images/pedes.3gpp")
 
#Loop once video is openes
while cap.isOpened():
    ret, frame = cap.read()
    
    frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow("Pedestrains", frame)
        
    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()
