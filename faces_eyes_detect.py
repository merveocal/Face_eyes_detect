import cv2 as cv
import numpy as np

video = cv.VideoCapture(0)

face_detect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml' )
while True:
    ret, frame = video.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale( gray, 
     scaleFactor=1.3, 
     minNeighbors=5, 
     minSize=(15, 15))
    
    # print(faces)  #yüzün koordinatlarını alabilirim
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y),(x+w, y+h), (255,0,0),3)

    
    
    
    eyes = eye_detect.detectMultiScale( gray, 
     scaleFactor=1.3, 
     minNeighbors=5, 
     minSize=(15, 15))
    
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(frame, (ex,ey),(ex+ ew, ey+eh),(15,48,185),3)
        
    cv.imshow("kamera", frame)
        
    if cv.waitKey(1) & 0XFF == ord('q'):
        print("Kamera sonlandırıldı")
        break
    

    
    
video.release()
cv.destroyAllWindows()