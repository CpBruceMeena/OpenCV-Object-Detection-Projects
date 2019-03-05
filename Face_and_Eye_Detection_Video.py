#Creating live face and eye detection
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier("C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/HaarCascade_XML files/haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/HaarCascade_XML files/haarcascade_eye.xml")

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
       print("No face detected")
    else:
        for (x, y, w, h) in faces:
            x = x - 50
            y = y - 50
            w = w + 50
            h = h + 50
            
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_classifier.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
        
        return img
    
cap = cv2.VideoCapture(cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    cv2.imshow("Our face detection ", face_detector(frame))
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()        