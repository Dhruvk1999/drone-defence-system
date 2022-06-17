'''
this is only for capturing the first frame
captures the frame and saves it to the directory
all paths have been fixed
'''
#imports 
import cv2
import numpy as np
import re


#coor=(left_x,top_y,width,height)
# Read video
cap = cv2.VideoCapture(0)
success,image = cap.read()
count = 0
if success:
    cv2.imwrite("./darknet/frame%d.jpg" % count, image)     # save frame as JPEG file   
    print("image is printed")   
    success,image = cap.read()
    print('Read a new frame: ', success)
    
cap.release()
cv2.destroyAllWindows()




