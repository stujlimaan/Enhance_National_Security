import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.createLBPHFaceRecognizer();
#recognizer=LBPHFaceRecognizer_create();
path='dataSet'

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #print(imagePaths)
    faces=[]
    IDs =[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    #return IDs,faces
    return np.array(IDs),faces
Ids,faces=getImagesWithID(path)
recognizer.train(faces,Ids)
#recognizer.train(faces,np.array(ids))
recognizer.save('recognizer/trainner.yml')
cv2.destroyAllWindows()








