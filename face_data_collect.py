k# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 23:47:07 2019

@author: raushan
"""
#Write a Python script that captures imaes from webcam video stream
#Extracts all Faces from the image frame (using haarcascades)
#Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to genrate training data

#import statements

import cv2
import numpy as np

#Init web Cam
cap = cv2.VideoCapture(0)

#Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
dataset_path = "./data/"
file_name = input("Enter the name of person")

while True:
    ret, frame = cap.read()
    
    if ret == False:
        continue
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    if len(faces) == 0:
        continue
    
    faces = sorted(faces, key=lambda f:f[2]*f[3])
    
    #pick the last face (because it has largest area)
    
    for face in faces[-1:]:
        #draw bounding box or the rectange
        
        x,y,w,h = face
        cv2.rectangle(gray_frame, (x,y), (x+w, y+h), (123, 88, 202), 2)
        
        #extract (crop out the required face): region of interest
        
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))
        face_data.append(face_section)
        print(len(face_section))
        
    #cv2.imshow("Frame", frame)
    cv2.imshow("Gray_Frame", gray_frame)
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#convert face data list into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("data saved successfully!!!")

cap.release()
cv2.destroyAllWindows()
        