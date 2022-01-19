#Import the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import zipfile
import os


def Load_dataset():
    dataset_path = 'Dataset_2'
    alphabet_list = os.listdir(dataset_path)
    print(alphabet_list)

    dataset = []
    for alphabets in alphabet_list:
      # Get all the file names
      all_filenames = os.listdir(dataset_path + '/' + alphabets)
      #print(all_filenames)

      # create filepaths for all filenames and add to the dataset along with alphabet category.
      for filename in all_filenames:
        dataset.append((alphabets, str(dataset_path+'/'+alphabets)+'/'+filename))

    #len(dataset)

    # Build a dtaframe.
    print("Building dataframe.....")
    isl_df = pd.DataFrame(data=dataset, columns=['Alphabet', 'Image'])
    print(isl_df.head())

    # Loading images.
    print("Loading images....")
    path = dataset_path + '/'
    im_size = 128
    images = []
    labels = []

    count = 0
    for item in alphabet_list:
        alphabet_folder = path + str(item)  # entered in the respective folders of each alphabet.
        filenames = [i for i in os.listdir(alphabet_folder)]

        for f in filenames:
            img = cv2.imread(alphabet_folder + '/' + f)  # read the image as an array.
            img = cv2.resize(img, (im_size, im_size))
            img= img / 255

            images.append(img)
            labels.append(item)
            count+=1
            if count%500==0:
                print(' images loaded : ',count)

    print("Converting to numpy array")
    images = np.array(images)
    labels = np.array(labels)
    print("Shape of images",images.shape)
    print("Shape of labels",labels.shape)

    print("Encoding Y ..........")
    Y = isl_df['Alphabet'].values
    y_labelencoder = LabelEncoder()
    Y = y_labelencoder.fit_transform(Y)
    #print(Y[:5])

    Y = Y.reshape(-1,1)
    onehotencoder = OneHotEncoder(categories = 'auto')
    Y = onehotencoder.fit_transform(Y)

    print("Splitting into train test sets...")
    images, Y = shuffle(images, Y, random_state=1)
    X_train, X_test, Y_train, Y_test = train_test_split(images, Y, test_size=0.1, random_state=415)

    # Checking the shapes of all train test sets
    print("Shape of X_Train: ",X_train.shape)
    print("Shape of X_test: ",X_test.shape)
    print("Shape of Y_train: ",Y_train.shape)
    print("Shape of Y_test: ",Y_test.shape)

    return X_train, X_test, Y_train, Y_test
