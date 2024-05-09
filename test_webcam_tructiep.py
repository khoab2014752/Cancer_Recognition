import cv2
import PIL.Image,PIL.ImageTk
from tensorflow import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import numpy as np
import tensorflow  as tf
from keras.preprocessing import image
from numpy import argmax

def majority_element(num_list):
    idx, ctr = 0, 1
        
    for i in range(1, len(num_list)):
        if num_list[idx] == num_list[i]:
            ctr += 1
        else:
            ctr -= 1
            if ctr == 0:
                idx = i
                ctr = 1
        
    return num_list[idx]


cap=cv2.VideoCapture(0)

model=keras.models.load_model("lenet[0].keras", compile=False)
categories = ['healthy', 'malignant', 'benign', 'other']
dem=0
Majority=[]
Ten=""
while True:
    ret , img=cap.read()
    
    im_pil = PIL.Image.fromarray(img)
    image = img
    image = cv2.resize(image, (16, 16))  # .flatten()
    
    im_resized = im_pil.resize((16, 16))
    img_array = tf.keras.utils.img_to_array(im_resized)
    image = np.array(image, dtype="float") / 255.0            
    image=np.expand_dims(image, axis=0)
    # img_preprocessed=preprocess_input(img_batch)

    print(image.shape)
    pred=model.predict(image)
    Res=argmax(pred,axis=1)
    # print(pred)
    Majority.append(Res[0])
    dem = dem +1
    print(dem)
    if dem > 30:
        dem=0
        Ten="{0}".format(categories[majority_element(Majority)])
        Majority=[]
    cv2.putText(img,Ten, (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        

    Result_Text="{0}({1})".format(categories[Res[0]],round(pred[0][Res[0]]*100,2))
    print("----------------------------------------")
    print("KQ:",Result_Text)
    print("----------------------------------------")
    
    cv2.imshow("camera",img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


