import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE



# ========================duong dan======================================

# path="./vtho/"
# categories = ['BD_KT','DT_mu','vtho_1']

# path="./tay/"
# categories = ['BD_KT','DT_mu', 'tay_1', 'tay_2','tay_3']

# path="./chan/"
# categories = ['BD_KT', 'chan_1', 'chan_2','DT_mu']

path="D://Khoa/Cancer_Recognition/Types/"
categories = ['healthy', 'malignant', 'benign', 'other']

# path="./bung/"
# categories = ['BD_KT', 'bung_1', 'bung_2','bung_3','DT_mu']

# path="./toanthan/"
# categories = ['BD_KT','DT_mu','toanthan_1', 'toanthan_2','toanthan_3']

# path="./nhay/"
# categories = ['BD_KT','DT_mu', 'nhay_1', 'nhay_2']

# path="./dieuhoa/"
# categories = ['BD_KT', 'dieuhoa_1', 'dieuhoa_2','DT_mu']

# =============================resize kich thuoc anh=================================

data = []#dữ liệu
labels = []#nhãn
imagePaths = []
HEIGHT = 16
WIDTH = 16
N_CHANNELS = 3

# ===========================lay ngau nhien anh===================================

for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k]) 

import random
random.shuffle(imagePaths)
# print(imagePaths[:10])

# =======================tien xu ly=======================================

for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    
    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    data.append(image)
    label = imagePath[1]
    labels.append(label)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

plt.subplots(4,4)
for i in range(16):
    plt.subplot(4,4, i+1)
    plt.imshow(data[i])
    plt.axis('off')
    plt.title(categories[labels[i]])
# plt.show()

# ============================chia tap dl==================================


# Early stopping

#     earlystop_callback = tf.keras.callbacks.EarlyStopping(
#     monitor='accuracy',  # Monitor validation loss
#     min_delta=0.001,    # Minimum change to be considered an improvement
#     patience=10,          # Stop training if no improvement for 10 epochs
#     mode='max'           # 'min' for loss, 'max' for accuracy
# )



# ===========================huan luyen===================================


#===
EPOCHS = 200
BS=32
Acc=[]
Precision = []
Recall = []
F1 = []
index=0
learning_rate=0.001
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',  # Monitor validation loss
    min_delta=0.001,    # Minimum change to be considered an improvement
    patience=10,          # Stop training if no improvement for 10 epochs
    mode='min'           # 'min' for loss, 'max' for accuracy
)
while index<1:
    
    

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=30)# random_state=30)
    trainY = keras.utils.to_categorical(trainY, len(categories))
    class_names = categories
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(WIDTH, HEIGHT, 3)))
    model.add(MaxPooling2D(strides=2))
    model.add(Convolution2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(strides=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(len(class_names), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])

    print(model.summary())

    model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, verbose=1, callbacks=[earlystop_callback])

    model.save('lenet[' + str(index)  + '].keras')
    # ==========================kiem tra su dung cua mo hinh====================================

    from numpy import argmax
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

    pred = model.predict(testX)
    predictions = argmax(pred, axis=1) # return to label

    cm = confusion_matrix(testY, predictions)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Model confusion matrix of' +str(i+1))
    fig.colorbar(cax)
    ax.set_xticklabels([''] + categories)
    ax.set_yticklabels([''] + categories)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(i, j, cm[j, i], va='center', ha='center')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    

    print("Run:",index+1)
    print("\n")
    accuracy = accuracy_score(testY, predictions)
    Acc.append(accuracy)
    
    print("Accuracy : %.2f%%" % (accuracy*100.0))
    print("\n")
    # ----------------------------------------------

    recall= recall_score(testY, predictions,average='weighted')
    Recall.append(recall)
    print("Recall :%.2f%%" % (recall*100))
    print("\n")
    # ----------------------------------------------

    precision = precision_score(testY, predictions,average='weighted')
    Precision.append(precision)
    print("Precision : %.2f%%" % (precision*100.0))
    print("\n")
    # ----------------------------------------------

    f1 = f1_score(testY, predictions,average='weighted')
    F1.append(f1)
    print("F1 : %.2f%%" % (f1*100.0))
    print("\n")

    print('Acc:',Acc)
    print('Pre:', Precision)
    print('Recall:',Recall)
    print('F1:', F1)
    print("Current index: ",index)
    index= index+1
plt.show()
  
print("Max index of acc",Acc.index(max(Acc))) 
print("Max index of pre",Precision.index(max(Precision))) 
print("Max index of recall",Recall.index(max(Recall))) 
print("Max index of f1",F1.index(max(F1))) 

# ==============================dua anh vao kiem tra================================

# from numpy import argmax
# import PIL
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image
# img_path="./nhan480.jpg"

# img=image.load_img(img_path,target_size=(32,32))
# img_array=image.img_to_array(img)
# from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
# img_batch=np.expand_dims(img_array, axis=0)
# img_preprocessed=preprocess_input(img_batch)

# pred=model.predict(img_preprocessed)
# Res=argmax(pred,axis=1)
# print(pred)

# plt.imshow(img)
# plt.show()
# print(categories[Res[0]],pred[0][Res[0]]*100)


