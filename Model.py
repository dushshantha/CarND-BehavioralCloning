import csv
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout



'''This function will read the CSV driving log files
    This accepts a list of log files and read each line into the memory.
'''
def readData(log_files):
    lines = []
    for file in log_files:
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
    return lines


#print(len(lines))

''' This function will print an image using Matplot lib on screen
'''
def print_Image(img):
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.show()

'''This function will display the Histogram of how the training data
    is distributed. This will tell you how balanced the data is'''
def histogram():
    (n, bins, patches) = plt.hist(Y_train, bins=20)
    print(bins)
    for item in n:
        print(int(item))

    #print(patches)
    plt.show()

'''This function will preprocess the image.
    I am converting the image to BGR
    Then I crop the image to remove all the distractions and
    keep only the road'''
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image[60:140, 0:320]
    return image


'''This function is where I read the images and steering angles from the lines
    in the CSV and prepare the test data.'''
def read_images(lines):
    images = []
    stearing_angles = []

    for record in lines:
        # Center image
        orig_path = record[0]
        new_path = './IMG/' + orig_path.split('/')[-1]
        image = cv2.imread(new_path)
        image = preprocess(image)
        images.append(image)
        angle = float(record[3])
        stearing_angles.append(angle)
        #print_Image(image)

        images.append(np.fliplr(image))
        stearing_angles.append(angle * -1)
        #print_Image(image)

        # Left Image
        orig_path = record[1]
        new_path = './IMG/' + orig_path.split('/')[-1]
        image = cv2.imread(new_path)
        image = preprocess(image)
        images.append(image)
        angle = float(record[3]) + 0.2
        stearing_angles.append(angle)

        images.append(np.fliplr(image))
        stearing_angles.append(angle * -1)

        # Right Image
        orig_path = record[2]
        new_path = './IMG/' + orig_path.split('/')[-1]
        image = cv2.imread(new_path)
        image = preprocess(image)
        images.append(image)
        angle = float(record[3]) - 0.2
        stearing_angles.append(angle)

        images.append(np.fliplr(image))
        stearing_angles.append(angle * -1)

    return np.array(images), np.array(stearing_angles)


'''This function is where I experimented with a Lenet architecture 
    that I used for Traffic Sign Classifier with few modifications'''
def train_model_LENET(modefile):
    model = Sequential()
    model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=X_train[0].shape))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train,Y_train,validation_split=0.25, shuffle=True, nb_epoch=2, batch_size=200)

    model.save(modefile)
    print("Model Saved...")


'''This function is where I adapted the NVIDIA model architecture 
    (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) 
    This was the final model I used to successsfully train the model.'''
def train_model_NVIDIA(modefile):
    model = Sequential()
    model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=X_train[0].shape))
    model.add(Convolution2D(24,5,5,activation='relu',border_mode='valid'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(48, 5, 5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train,Y_train,validation_split=0.25, shuffle=True, nb_epoch=2, batch_size=100)

    model.save(modefile)
    print("NVIDIA Model Saved...")


'''The main function where I execute the process'''

if __name__ == '__main__':

    # Read the logs
    # log_files = ["./driving_log_foreward.csv", "./driving_log_backwords.csv"]
    log_files = ["./driving_log_regular.csv", "./driving_log_recovery1.csv", "./driving_log_recovery2.csv"]
    lines = readData(log_files)

    X_train, Y_train = read_images(lines)

    print(X_train[0].shape)
    print(Y_train.shape)

    print(Y_train.shape[0] / 20.0)

    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    #print_Image(image)
    #histogram()

    # train_model_LENET('model_LENET.h5')
    train_model_NVIDIA('model_NVIDIA.h5')