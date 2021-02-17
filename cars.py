# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:10:58 2021

@author: Antoine Chatry
"""
import cv2

#Image
img_file = 'car.jpg'
#video = cv2.VideoCapture('MotoCycle.mp4')
video = cv2.VideoCapture('Tesla.mp4')
#Pre-trained car classifier
classifier_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

#Create classifiers
car_tracker = cv2.CascadeClassifier(classifier_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

#Run forever until car stops or something
while True:
    #Read current frame
    (read_successful, frame) = video.read()
    #img = cv2.imread(img_file)
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    #detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
    #Draws Rectangles around cars
    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
    
    for(x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    #Display image with faces spotted
    cv2.imshow('Cars Detector', frame)
    
    #Don't auto close (listen for a key press)
    key = cv2.waitKey(1)
    
    #Stop if Q key is pressed
    if key==81 or key==113:
        break

#Release video capture
video.release()






