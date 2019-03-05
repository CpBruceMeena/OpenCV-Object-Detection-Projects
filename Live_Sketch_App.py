"""
Creating Live Sketch App

"""
import cv2
import numpy as np

#Our sketch generating function
def sketch(image):
    
    #Converting image to gray image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Converting that gray image to Gaussian Blur
    #Clean up image and clears all the noise in the image
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    
    #Extract Edges 10 and 70 are the thresholds
    #Canny is black background with white edges(You can vary it and see what happens)
    canny_edges = cv2.Canny(img_gray_blur, 5, 30)
    
    #Do an invert binarize the image
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask

#Initialize webcam, cap is the object provided by VideoCapture
#It contain a boolean indicating if it was successful (ret)
#It also contains the images collected form the webcam (frame)

cap = cv2.VideoCapture(0)

while True:
    # "Ret" will obtain return value from getting the camera frame,
    # either true of false
    
    #It pulls an image for webcam
    ret, frame = cap.read()
    
    cv2.imshow("Our live sketcher", sketch(frame))

    #earlier the wait key was zero, it means pressing any key will stop it
    #but now it's one means pressing a particular key will stop the windows
    if cv2.waitKey(1) == 13:  #13 is the enter key
        break

#Releases the camera and close windows
cap.release()
cv2.destroyAllWindows()
