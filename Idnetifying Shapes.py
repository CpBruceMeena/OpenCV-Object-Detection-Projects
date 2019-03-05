#Mini Project Shape Making
import cv2
import numpy as np

#to load an image 
image = cv2.imread('C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python_Projects/CV2 Learning Codes/Required Images/shapes.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Original Image", image)
cv2.waitKey(0)

#Converting to thershold
ret, thresh = cv2.threshold(gray, 127, 255, 1)

#Extracting to contours
extra, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#Sorting the contours so that we can remove the largest by default image frame
sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)

contours = sorted_contours[1:]
print(len(contours))

for cnt in contours:
    print("hello")
    #Get approximate polygons
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    
    if len(approx)  == 3:
        shape_name = "triangle"
        cv2.drawContours(image, [cnt], 0, (0, 255, 0), -1)


    #tFind contour center to place the text at the center       
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        
    #We are subtracting 50 because the point we get is bottom right corner
        cv2.putText(image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
    elif len(approx) == 4:

        #this is done to check whether it is a square or rectangle
        #we are checking between the height and width

        x,y,w,h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
 
        
        if abs(w-h) <= 3:
            shape_name = 'Square'
            cv2.drawContours(image, [cnt], 0, (125, 125, 255), -1)
            cv2.putText(image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        
        else:
            shape_name = 'Rectangle'
            cv2.drawContours(image, [cnt], 0, (0, 0, 255), -1)
            cv2.putText(image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1) 

    
    elif len(approx) == 10:
        shape_name = 'star'
        cv2.drawContours(image, [cnt], 0, (255, 255, 0), -1)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.putText(image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)


    elif len(approx) >= 15:
        shape_name = 'Circle'
        cv2.drawContours(image, [cnt], 0, (0, 255, 255), -1)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.putText(image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)


    cv2.imshow("Identifying shapes", image)
    cv2.waitKey(0)

#when you have seen all the windows press ESC at the last to destroy all the windows
#the waitKey(0) means it is waiting for ESC key
#waitKey(1) == 13 means press ENTER
#This closes all windows
cv2.destroyAllWindows()

