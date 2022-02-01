import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, InputLayer
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model


#reading the csv data
data = pd.read_csv("C:/Users/User/Desktop/age_gender.csv")
df = data.drop('img_name', axis=1)
print(df)
columns = ["age", "gender", "ethnicity"]
y = df.drop("pixels", axis=1)
X = df.drop(columns, axis=1)


num_pixels = len(X['pixels'][0].split(" "))
img_height = int(np.sqrt(len(X['pixels'][0].split(" "))))
img_width = int(np.sqrt(len(X['pixels'][0].split(" "))))
#the shape of our data  is  (2304 48 48)
print(num_pixels, img_height, img_width)

X = pd.Series(X['pixels'])
X = X.apply(lambda x:x.split(' '))
X = X.apply(lambda x:np.array(list(map(lambda z:np.int(z), x))))
X = np.array(X)
X = np.stack(np.array(X), axis=0)


X = X.reshape(-1, 48, 48, 1)
print("X shape: ", X.shape)

y["age"].unique()

y["age"] = pd.cut(y["age"],bins=[0,3,18,45,64,116],labels=["0","1","2","3","4"])  # we are dividing our age column into 6 groups

age_matrix = np.array(y['age'])
gender_matrix = np.array(y['gender'])
ethnicity_matrix = np.array(y['ethnicity'])
age = to_categorical(age_matrix, num_classes = 5)
gender = to_categorical(y["gender"], num_classes = 2)
ethnicity = to_categorical(ethnicity_matrix, num_classes = 5)
# print(age, gender, ethnicity)

datagen = ImageDataGenerator(
        featurewise_center = False,
    # set input mean to 0 over the dataset
       samplewise_center = False,
    # set each sample mean to 0
       featurewise_std_normalization = False,
    # divide inputs by std of the dataset
       samplewise_std_normalization=False,
    # divide each input by its std
       zca_whitening=False,
    # dimesion reduction
       rotation_range=5,
    # randomly rotate images in the range 5 degrees
       zoom_range = 0.1,
    # Randomly zoom image 10%
       width_shift_range=0.1,
    # randomly shift images horizontally 10%
       height_shift_range=0.1,
    # randomly shift images vertically 10%
       horizontal_flip=False,
    # randomly flip images
        vertical_flip=False  # randomly flip images
)
datagen.fit(X)
# Ethnicity
X_train_ethnicity, X_test_ethnicity, y_train_ethnicity, y_test_ethnicity = train_test_split(X,ethnicity, test_size=0.3, random_state=42)

# Gender
X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(X, gender, test_size=0.3, random_state=42)

# Age
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X, age, test_size=0.3, random_state=42)
print(X_train_ethnicity.shape, X_train_gender.shape, X_train_age.shape)

def my_model(num_classes, activation, loss):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(num_classes, activation=activation))

    model.compile(optimizer='Adam',
                  loss=loss,
                  metrics=['accuracy'])
    return model

early_stopping = EarlyStopping(patience=10,
                               min_delta=0.001,
                               restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                           patience = 2,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr = 0.00001)

epochs = 500  # for better result increase the epochs
batch_size = 64

#model for ethnicity detection
model_ethnicity = my_model(5,"softmax",'categorical_crossentropy')
# history_ethnicity = model_ethnicity.fit(X_train_ethnicity, y_train_ethnicity, batch_size=batch_size,
#                               epochs = epochs, validation_data = (X_test_ethnicity,y_test_ethnicity), steps_per_epoch= X_train_ethnicity.shape[0] // batch_size, callbacks= [early_stopping, learning_rate_reduction])
#
#
#
# loss, acc = model_ethnicity.evaluate(X_test_ethnicity, y_test_ethnicity, verbose=0)
# print('Test loss: {}'.format(loss))
# print('Test Accuracy: {}'.format(acc))

model_ethnicity.save("model_ethnicity.h5")
model_ethnicity_saved = load_model("model_ethnicity.h5")

#model for age detection
model_age = my_model(5,"softmax",'categorical_crossentropy')
# history_age = model_age.fit(X_train_age, y_train_age, batch_size=batch_size,
#                               epochs = epochs, validation_data = (X_test_age,y_test_age),
#                             steps_per_epoch= X_train_age.shape[0] // batch_size,
#                             callbacks= [early_stopping,
#                             learning_rate_reduction])
#
# loss, acc = model_age.evaluate(X_test_age, y_test_age, verbose=0)
# print('Test loss: {}'.format(loss))
# print('Test Accuracy: {}'.format(acc))

model_age.save("model_age.h5")
model_age_saved = load_model("model_age.h5")


#model for gender detection
model_gender = my_model(2, "sigmoid", "binary_crossentropy")
# history_gender = model_gender.fit(X_train_gender, y_train_gender,
#                                  batch_size = batch_size,
#                                  epochs = epochs,
#                                  validation_data = (X_test_gender, y_test_gender),
#                                  steps_per_epoch = X_train_gender.shape[0] // batch_size, callbacks=[early_stopping,learning_rate_reduction])
#
# loss, acc = model_gender.evaluate(X_test_gender, y_test_gender, verbose=0)
# print("Test loss: {}".format(loss))
# print("Test Accuracy: {}".format(acc))

model_gender.save("model_gender.h5")
model_gender_saved = load_model("model_gender.h5")

#function for our gender detection model
def gender_detection(image):
    resized_image1 = resize(image, (48, 48, 1))
    predictions = model_gender_saved.predict(np.array([resized_image1]))
    list_index = [0, 1]
    x = predictions
    classes = ["male","female"]
    for i in range(1):
        for j in range(1):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return classes[list_index[0]]

#function for our age detection model
def age_detection(image):
    resized_image1 = resize(image, (48, 48, 1))
    predictions = model_age_saved.predict(np.array([resized_image1]))
    list_index = [0, 1, 2, 3, 4]
    classes = [0, 3, 18, 45, 64, 116]
    x = predictions
    for i in range(5):
        for j in range(5):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return classes[list_index[0]]

#function for our ethnicity detection
def ethnicity_detection(image):
    resized_image1 = resize(image, (48, 48, 1))
    predictions = model_ethnicity_saved.predict(np.array([resized_image1]))
    list_index = [0, 1, 2, 3, 4]
    classes = ["black", "White", "american indian", "Asian", "Native Hawaiian or Other Pacific Islander"]
    x = predictions
    for i in range(5):
        for j in range(5):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return classes[list_index[0]]






