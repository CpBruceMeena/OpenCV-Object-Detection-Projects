#Object Detection 
#Mini Project
"""
 We are taking a template and trying to figure out whether the vectors of the initial image matches with the
 vectors of the template and if they match, it implies that the object has been detected.
"""

import cv2
import numpy as np

def ORB_detector(new_image, image_template):
    #Function that compares image and the template
    #It then returns the number of ORB matches between them
    
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    
    #Create ORB detector with 1000 keypoints with a scaling pyramid of 1.2
    orb = cv2.ORB_create(1000, 1.2)
    
    #Detect keypoints of original image
    (kp1, des1) = orb.detectAndCompute(image1, None)
    
    #Detect keypoints of rotated image
    (kp2, des2) = orb.detectAndCompute(image_template, None)
    
    #Create Matcher
    #Note we're no longer using Flawboard matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
     
    #Do matching
    matches = bf.match(des1, des2)
     
    #Sort the matches based on distance, least distance
    matches = sorted(matches, key = lambda val: val.distance)
 
    return(len(matches))
    
#To start the video cam    
cap = cv2.VideoCapture(0)

#Take the image template that you want to look for
image_template = cv2.imread("C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/Required Images/chess.jpg", 0)    
    
while(True):

    #To capture the images in the video
    ret, frame = cap.read()
    
    #Set height and width of webcam frame
    height , width = frame.shape[:2]
    
    #Define ROI Box Dimension 
    top_left_x = int(width/3)
    top_left_y = int((height/2)*(height/4))
    bottom_right_x = int((width/3)*2)
    bottom_right_y = int((height/2)-(height/4))
    
    #Draw rectangle window for our region of interest
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, 3)
    
    #Crop window of observation we defined above
    cropped = frame[bottom_right_y:top_left_y, top_left_x:bottom_right_x]
    
    #Flip frame horizontally
    frame = cv2.flip(frame, 1)
    
    #Get number of ORB matches
    matches = ORB_detector(cropped, image_template)
    
    #Display states string showing the correct no. of matches
    output_string = "matches : "+ str(matches)
    cv2.putText(frame, output_string, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (250, 0, 150),2)
    
    #Our threshold to intricate object detection
    #for any image or lightening condition you may need to experiment
    #Note: the ORB detector to get the top 1000 matches, 350 is essentially a min. 35% match
    threshold = 350
    
    #It matches exceeds threshold, then object has been detected
    if matches > threshold:
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, 3)
        cv2.putText(frame, "Object Found ", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    
    cv2.imshow("Object Detector using ORB", frame)
    
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
