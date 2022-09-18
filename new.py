import dlib
import cv2
import face_recognition
ad_img = face_recognition.load_image_file("opencv_frame_1.png")
ad_img=cv2.cvtColor(ad_img,cv2.COLOR_BGR2RGB)
as_img=face_recognition.load_image_file('opencv_frame_0.png')
as_img=cv2.cvtColor(as_img,cv2.COLOR_BGR2RGB)

ad_FaceLoc=face_recognition.face_locations(ad_img)[0]
encode_ad = face_recognition.face_encodings(ad_img)[0]
cv2.rectangle(ad_img,(ad_FaceLoc[3],ad_FaceLoc[0]),(ad_FaceLoc[1],ad_FaceLoc[2]),(255,0,0),2)


as_FaceLoc=face_recognition.face_locations(as_img)[0]
encode_as=face_recognition.face_encodings(as_img)[0]
cv2.rectangle(as_img,(as_FaceLoc[3],as_FaceLoc[0]),(as_FaceLoc[1],as_FaceLoc[2]),(255,0,0),2)

result=face_recognition.compare_faces([encode_ad],encode_as)
face_distance=face_recognition.face_distance([encode_ad],encode_as)
print(result,face_distance)
cv2.putText(as_img,f'{result[0]}   {round(face_distance[0],2)}',(50,50),0,0.8,(0,0,255),2)

cv2.imshow("Ashish",as_img)
cv2.imshow('Adrash',ad_img)
cv2.waitKey(0)
