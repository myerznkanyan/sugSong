import numpy as np
import pandas as pd
import tensorflow
import keras
from tensorflow.keras import layers
import tensorflow as tf
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import PIL
from numpy import asarray
from skimage.transform import resize
import os
from keras.models import load_model

plt.style.use("fivethirtyeight")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_path = Path('C:/Users/User/Desktop/archive')

list1 = []
filepaths = list(image_path.glob(r'**/*.jpg'))
for i in filepaths:
    image = PIL.Image.open(i)
    image_array = np.array(image)
    list1.append(image_array)

x_train = np.array(list1)
print(x_train.shape)
x_train = x_train.reshape(28709, 48,48,1)

# makeing our data numpy array
filepaths = pd.Series(list(image_path.glob(r'**/*.jpg')), name="filepaths")
mood = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='mood')
moodi = pd.get_dummies(mood)
classes = ["angry", "disgust", "fear", "happy", "netural", "sad", "surprise"]
numpy_array = moodi.to_numpy()
y_train = numpy_array

# look at the data types of the variables
print(type(x_train))
print(type(y_train))

# get the shapes of the arrays
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# show the image as a picture
index = 0
img = plt.imshow(x_train[index])
plt.show()

# get the image label
print("the image label is:", y_train[index])

# get the image classification
# print the image class
print("the image class is", classes[y_train[index][0]])

# normalize the pixels to be values between 0 and 1
x_train = x_train / 255

# working with test data
test_path = Path('C:/Users/User/Desktop/test')
list0 = []
filepaths0 = list(test_path.glob(r'**/*.jpg'))
for i in filepaths0:
    image = PIL.Image.open(i)
    image_array = np.array(image)
    list0.append(image_array)
x_test = np.array(list0)
x_test.shape
x_test = x_test.reshape(7178, 48,48,1)

filepaths0 = pd.Series(list(test_path.glob(r'**/*.jpg')), name="filepaths")
mood_test = pd.Series(filepaths0.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='mood')
mood0 = pd.get_dummies(mood_test)
numpy_array = mood0.to_numpy()
y_test = numpy_array

print(type(x_test))
print(type(y_test))
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
x_test = x_test / 255
x_test = x_test.reshape(7178, 48, 48, 1)

# Add the first layer
model = keras.models.Sequential()
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(7, activation='softmax'))

# compile the model
model.compile(loss = "categorical_crossentropy",optimizer = "adam",metrics = ["accuracy"])

# train the model
# hist = model.fit(x_train,y_train,
#                  batch_size=256,
#                  epochs =10,
#                  validation_split=0.2)
#
# # Evaluate the model using the test data set
# model.evaluate(x_test,y_test)


model.save("mood.h5")
mood_saved = load_model("mood.h5")

def doit(image):
     resized_image1 = resize(image, (48,48,1))
     predictions = mood_saved.predict(np.array([resized_image1]))
     list_index = [0,1,2,3,4,5,6]
     x = predictions
     for i in range(7):
          for j in range(7):
               if x[0][list_index[i]]>x[0][list_index[j]]:
                   temp = list_index[i]
                   list_index[i]=list_index[j]
                   list_index[j]=temp
     return classes[list_index[0]]




