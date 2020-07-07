import tensorflow as tf
import cv2

CATEGORIES = ['Dog', 'Cat']

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE,IMG_SIZE,1)

model = tf.keras.models.load_model('catsvdogs-convs-3-nodes-[64]-dense-0-1594138760')

prediction = model.predict([prepare('107.jpg')])

print(CATEGORIES[int(prediction[0][0])])