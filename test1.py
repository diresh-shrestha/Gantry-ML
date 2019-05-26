from __future__ import absolute_import, division, print_function, unicode_literals
#os to load directories and cv2 for preprocessing images
import os, cv2
import numpy as np
import matplotlib.pyplot as plt

#shuffles the dataset
from sklearn.utils import shuffle
#splits the dataset into training and test
from sklearn.model_selection import train_test_split

#Keras apparently does not handle low-level operations such as tensor products, convolutions, etc.
#so it relies on other tensor manipulation libraries to serve as the backend engine of keras
#backend as K makes it compatible with both Tensorflow and Theano
from keras import backend as K



import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, models


from keras.utils import np_utils
#from keras.layers.normalization import BatchNormalization



#defining the size of the images after processing
img_rows = 250
img_columns = 250
#our image will processed in RGB so channel is 3.
num_channels = 3
#we have 2 classes, 'Yes' or 'No'
num_classes = 2
num_epoch = 20

#our labels
labels = {'Yes':1, 'No': 0}

#creating two empty arrays.
img_data_list = []
labels_list = []

#giving the directory 
#PATH = 'D:\Machine Learning\Gantry_vision'
data_path = 'D:\Machine Learning\Gantry_vision\data'
#os.listdir lists all the directories inside data_path
data_dir_list = os.listdir(data_path)
#print (data_dir_list)

#for loop to load all the images
for dataset in data_dir_list:
    #lists all the images inside each folder
    img_list = os.listdir(data_path + '/' + dataset)
    print ('Loaded the images of the dataset-' + '{}\n'.format(dataset))
    label = labels[dataset]
    #preprocessing the images. 
    for img in img_list:
        #read each image
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        #resize each image into 250*250
        #input_img_resize = cv2.resize(input_img,(250,250))
        #append the images into img_data_list array
        img_data_list.append(input_img)
        #append the labels                 
        labels_list.append(label)

''' #optional code to add Gaussian blur to the images to reduce noise       
    original = img_data_list[20]
    no_noise = []
    for i in range(len(img_data_list)):
        blur = cv2.GaussianBlur(img_data_list[i], (5, 5), 0)
        no_noise.append(blur)

def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()

image = no_noise[20]
display(original, image, 'Original', 'Blurred')
 ''' 
#convert img_data_list into an array using np      
img_data = np.array(img_data_list)
#convert it into float32
img_data = img_data.astype('float32')
#rescale the matrix by dividing by 255 so every datum inside the matrix is in the range of 0 to 1.
img_data /= 255
print (img_data.shape)

#same thing but to labels
label = np.array(labels_list)
print(np.unique(label,return_counts=True))
#Y contains the labels using np_utils
Y = np_utils.to_categorical(label, num_classes)

#assigning x as the data and y as the labels and shuffling them
x,y = shuffle(img_data,Y, random_state=2)
#splitting the data using train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


input_shape = img_data[0].shape
print (input_shape)
 
#CNN layers   
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
#model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

#fit the model
hist = model.fit(X_train, y_train, batch_size=16, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))



train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

#plotting training loss vs validation loss
plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#plotting training accuracy vs validation accuracy
plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

