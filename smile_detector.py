#Smile recognition

#Import library
import cv2

#Loading cascades
face_cascade = cv2.CascadeClassifier('F:\\CS STUFF\\Udemy - computer-vision-a-z\\03 Module 1 - Face Detection with OpenCV\\Computer Vision A-Z\\Module 1 - Face Recognition\\haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('F:\\CS STUFF\\Udemy - computer-vision-a-z\\04 Homework Challenge - Build a Happiness Detector\\haarcascade_smile.xml')

#Defining a function that will do the detection
def detect(gray, frame):
    #1.3 is the scale factor
    #5 is the minimun neighbour, increasing the min neighbour value will improve the accuracy of the detection.  
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in faces:
        
        # writing the upper left corner and lower right coordinate, then color, and then width of the line.
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        #the eyes will be detected in the gray image and then the rectangle will be drawn in the original image
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
         
        #roi_gray is the region of interest where we'll look for the eye.
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 4)
        for (sx, sy, sw, sh) in smile:
            
            #roi_color is the image where we'll draw the rectangle
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    
    return frame

#Doint some face recognition with the webcam
# 0 is the code for the internal webcam for external webcam you should use 1
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
    