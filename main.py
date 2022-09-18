import cv2
import os
from mtcnn import MTCNN
# import  streamlit as st
from PIL import Image
import pickle
import cv2
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity

model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

detector = MTCNN()


# load Image and detect the face
sample_img = cv2.imread('pic1.jpg')
r = detector.detect_faces(sample_img)
x, y, width, height = r[0]['box']
face = sample_img[y:y + height, x:x + width]

image = Image.fromarray(face)
image = image.resize((224, 224))
face_array = np.asarray(image)
face_array = face_array.astype('float32')
expanded_img = np.expand_dims(face_array, axis=0)
preprocess_img = preprocess_input(expanded_img)
result = model.predict(preprocess_img).flatten()
print(result.shape)


