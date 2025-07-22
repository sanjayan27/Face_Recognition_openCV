#Importing required libraries
import numpy
import os
import cv2
import imutils
import math

#initialize the dataset path
datasets = "Datasets"

#initializing the values
(images, labels, names, id) = ([], [], {}, 0)

#using loop for getting all sub directories from the main folder 
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectPath = os.path.join(datasets,subdir)
        for fileNames in os.listdir(subjectPath):
            path = subjectPath + "/" + fileNames
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id += 1
        
#converting a single array using numpy array
(images,labels) = [numpy.array(lis) for lis in [images,labels]]

#assigning width and height for sending the face frame to model for predicting purpose
(width, height)  = (100,140)

#loading the haar cascade frontal face algorithm
haar_cascade_algorithm = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(haar_cascade_algorithm)

#initializing the path for video to predict
video = "Assets/all.mov"
video2 = "elon.mp4"

#loading the fisherface recognizer algorithm to predict the face using coordinates
model = cv2.face.FisherFaceRecognizer_create()

print("model is now training.... please wait")

#training the loaded model using images and labels 
model.train(images,labels)

print("model trained successfully")

#initializing the camera
camera = cv2.VideoCapture(video2)

while True:
    success,frame1 = camera.read()
    
    #if the video ended this will quitly exit the entire loop without throwing the error
    if not success:
        print("you video is completed")
        break
    else:
        frame = imutils.resize(frame1,700)
        
        #converting normal bgr frame into gray to predict more accurate
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        #getting multi scale coordinates using the haar cascade alg
        faces = haar_cascade.detectMultiScale(gray_frame,1.3,4)
        
        #looping the entire coordinates detected from the algrithm
        for (x,y,w,h) in faces:
            croped_image = gray_frame[y:y+h,x:x+w]
            resized_img = cv2.resize(croped_image,(width,height))
            prediction = model.predict(resized_img)
            cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),2)
            count = 0
            if(prediction[1]<800):
                cv2.putText(frame,f"{names[prediction[0]]}-{math.floor(prediction[1])}",(x-10, y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1)
                count = 0
            else:
                count+=1
                cv2.putText(frame,"Unknown",(x-10, y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
                if(count > 100):
                    cv2.imwrite("input.png",frame)
                    count = 0
    
    #showing the video 
    cv2.imshow("frame",frame)
    key = cv2.waitKey(10)
    if(key == ord("a")):
        break
    

camera.release()
cv2.destroyAllWindows()