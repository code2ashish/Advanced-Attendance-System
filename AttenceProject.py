import  cv2
import numpy as np
import face_recognition
import os
import datetime

path='ImageAttendece'
images=[]
ClassName=[]
mylist=os.listdir(path)
# print(mylist)

for cls in mylist:
    currImg=cv2.imread(f'{path}/{cls}')
    images.append(currImg)
    ClassName.append(os.path.splitext(cls)[0])
# print(ClassName)

def Encoder(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def markAttendece(name):
    with open('Attendence.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.datetime.now()
            dtSTR=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtSTR}')

encodeListKnown=Encoder(images)
# print(len(encodeListKnown))

cam=cv2.VideoCapture(0)

while True:
    succ,img=cam.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    imgS_FaceLoc = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS,imgS_FaceLoc)

    for encodeFace,faceLoc in zip(encodeCurrFrame,imgS_FaceLoc):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDistance=face_recognition.face_distance(encodeListKnown,encodeFace)

        matchIDX=np.argmin(faceDistance)
        if matches[matchIDX]:
            name=ClassName[matchIDX]
            print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),0,1,(255,255,255),1)
            markAttendece(name)
    cv2.imshow("webcam",img)
    cv2.waitKey(1)



