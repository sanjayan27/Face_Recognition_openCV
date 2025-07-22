import os
import cv2

#initializing paths
dataset = "Datasets"
folder_dataset = "Elon"
video = "elonMusk.mp4"

#if the path doesnt exits in the directory it will create a new path
path = os.path.join(dataset,folder_dataset)
if not os.path.isdir(path):
    os.mkdir(path=path) 
    
(width, height) = (100, 140)

#loading haar cascade frontal face algorithm 
algorithm = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(algorithm)

#initilizing camera
camera = cv2.VideoCapture(video)

#count forrun only 172 times
count = 1
while count < 173:
    _,frame = camera.read()
    
    #converting normal bgr frame into gray to predict more accurate
    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #getting multi scale coordinates using the haar cascade alg
    coordinates = haar_cascade.detectMultiScale(gray_image,1.3,4)
    
    #looping the entire coordinates detected from the algrithm
    for (x,y,w,h) in coordinates: 
        cv2.rectangle(frame,(x,y),(x + w , y+ h),(0,255,0),3)
        croped_img = gray_image[y:y+h,x:x+w]
        resize = cv2.resize(croped_img,(width,height))
        cv2.imwrite(f"{path}/{count}.png",resize)
        
    count += 1
    
    #showing the video 
    cv2.imshow("frame",frame)
    key = cv2.waitKey(10)
    if(key == 27):
        break
    
camera.release()
cv2.destroyAllWindows()
    
    
    