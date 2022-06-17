''' after running the huduki.py 
run this to track the drone if possible 
else jump to jamming
'''

import cv2
import numpy as np
import re

with open('../darknet/result.txt') as f:
    lines = f.readlines()
a=lines[-2]
print(lines)
b=a.split()
print(b)
left_x=int(b[3])
top_y=int(b[5])
width=int(b[7])
height=int(re.findall("^\d\d",b[9])[0])

coor=(left_x,top_y,width,height)
print(coor)

tracker = cv2.legacy.TrackerMedianFlow_create()
tracker_name = str(tracker).split()[0][1:]





# Read video
cap = cv2.VideoCapture(0)
success,image = cap.read()

ret, frame = cap.read()


# Special function allows us to draw on the very first frame our desired ROI
## replace this roi by the coordinates from the resultant image from the DL model resultant
roi = coor
# Initialize tracker with first frame and bounding box
ret = tracker.init(frame, roi)

#writer = cv2.VideoWriter('student_capture.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (500, 500))


while True:
    # Read a new frame
    ret, frame = cap.read()
    
    
    # Update tracker
    success, roi = tracker.update(frame)
    
    # roi variable is a tuple of 4 floats
    # We need each value and we need them as integers
    (x,y,w,h) = tuple(map(int,roi))
    
    # Draw Rectangle as Tracker moves
    if success:
        # Tracking success
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (0,255,0), 3)
        print(f'{p1} and {p2}')
    else :
        # Tracking failure
        cv2.putText(frame, "Failure to Detect Tracking!!", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

    # Display tracker type on frame
    cv2.putText(frame, tracker_name, (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3);
    
    #writer.write(frame)


    # Display result
    cv2.imshow(tracker_name, frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : 
        break
        
cap.release()
cv2.destroyAllWindows()

# use tracker 1 