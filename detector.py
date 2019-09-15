import cv2,os
import numpy as np
from PIL import Image
import pickle
import sqlite3

#recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.load('recognizer/trainner.yml')
recognizer.read('recognizer/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


def getProfile(id):
    conn=sqlite3.connect("facedatabase.db")
    cmd='SELECT * FROM people WHERE ID=' + str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

cam = cv2.VideoCapture(0)
id=0;

'''font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)'''
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 0, 0)

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,255),2)
        Ids, conf = recognizer.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(im, str(profile(1)), (x,y+h), fontface, fontscale, fontcolor)
            cv2.putText(im, str(profile[1]), (x,y-40), font, 2, (255,255,255), 3)
            cv2.putText(im, str(profile[1]), (x,y-10), font, 2, (255,255,255), 3)
            cv2.putText(im, str(profile[1]), (x,y+20), font, 2, (255,255,255), 3)
        '''if(conf>50):
            if(Ids==1):
                Ids="a"
            elif(Ids==2):
                Id="Something"
        else:
            Ids="Unknown" '''
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        font = cv2.FONT_HERSHEY_DUPLEX
       # cv2.putText(im,str(Id), (x,y+h),font, 1,fontcolor);
        #cv2.putText(im, str(Ids), (x,y-40), font, 2, (255,255,255), 3)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

