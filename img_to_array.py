import numpy as np
import os
import cv2
import random
import pickle

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except:
                pass

create_training_data()
random.shuffle(training_data)

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)
   
x = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y = np.array(y)

IMG_SIZE = 50

pickle_out = open('x_pickle','wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open('y_pickle','wb')
pickle.dump(y, pickle_out)
pickle_out.close()