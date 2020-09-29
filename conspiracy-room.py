# import the necessary packages
import numpy as np
import cv2
import pygame
from datetime import datetime, timedelta

pygame.init()

# config #

min_conspiracy_peeps = 2
stop_delay_seconds = 2

########## 

def start_music():
    pygame.mixer.music.set_volume(100)

def stop_music():
    global music_last_stopped_at

    stop_delay = datetime.now() - timedelta(seconds=stop_delay_seconds)

    if music_last_stopped_at < stop_delay:
        pygame.mixer.music.set_volume(0)
        music_last_stopped_at = datetime.now()
    else:
        pass

pygame.mixer.music.load("x.mp3")
pygame.mixer.music.set_volume(0)
pygame.mixer.music.play(-1)
music_last_stopped_at = datetime.now()

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)

    if len(boxes) >= min_conspiracy_peeps:
        start_music()
    else:
        stop_music()
    
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)