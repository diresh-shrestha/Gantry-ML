
import keras
import tensorflow as tf
from tensorflow.python.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, cv2
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import np_utils
import math
from keras.models import model_from_json
import sys
from functools import reduce 
from keras.backend import manual_variable_initialization
manual_variable_initialization(True)

def load_data(PATH = 'D:\Machine Learning\Gantry_vision\\New\lines_final.csv'):
    data = pd.read_csv(PATH, skipinitialspace=True)
    #the following was my start to trying to extract location data
    coord =  [0] * 530
    polar =  [0] * 530
    for s in range(530):
        coord[s] = [0,0,0,0]
        polar[s] = [0,0]
        pre_coord = data['Label'][s].split(',')
        #print("loop 1 - ",s)
        for i in range(4):
            #print("in 2nd loop - ", i)
            pre_coord[i] = pre_coord[i] + '}'
            hold = pre_coord[i][pre_coord[i].find(':')+1:pre_coord[i].find('}')]
            hold = float(hold)
            if i%2 != 0: coord[s][i] = hold/1920.0
            else: coord[s][i] = hold/2560.0
            coord[s][1] = 1- coord[s][1]
            coord[s][3] = 1-coord[s][3]


        m = (coord[s][3]-coord[s][1]) / (coord[s][2]-coord[s][0])
        theta = math.degrees(math.atan(m))
        #theta = math.degrees(math.atan2(coord[s][3]-coord[s][1], coord[s][2]-coord[s][0]))
        b, c = 1.0, coord[s][1]-m*coord[s][0]
        r = abs(c) / math.sqrt(m**2 + b**2)
        polar[s] = [theta/90, r]

    v = np.load('D:\Machine Learning\Gantry_vision\\New\\resized_img_array250x250.npy')
    return v,polar

def split_mse(label,pred):

    if len(label)!=len(pred) : return -1,-1
    n = float(len(pred))
    resid = np.transpose(label-pred)
    theta_mse = map(np.square,resid[0])
    theta_mse = math.sqrt(reduce(np.add,theta_mse)/n)
    r_mse = map(np.square,resid[1])
    r_mse = math.sqrt(reduce(np.add,r_mse)/n)
    
    return theta_mse,r_mse

def multi_mean(y_true, y_pred):
    resid=np.transpose(y_pred-y_true)
    
    return K.mean(y_pred-y_true)

def CNN(img_data,regress, prediction, train, opt, Pdim, cnn_num, f, num_epoch):

    x,y = img_data, regress
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=12345)
    X_train = np.array(X_train)
    X_train = np.delete(X_train,[0,1],3)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test = np.delete(X_test,[0,1],3)

    input_shape = X_test[0].shape
    pool_size = (Pdim, Pdim)


    #CNN layers   
    #xavier = tf.contrib.layers.xavier_initializer(seed=13)
    model = models.Sequential()
    cnn_size = 16
    model.add(layers.Conv2D(cnn_size, 5, activation='relu',  input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size))

    
    layers1 = cnn_num
    while(layers1 > 0):
        cnn_size *= 2
        model.add(layers.Conv2D(cnn_size, 5, activation='relu', ))
        model.add(layers.MaxPooling2D(pool_size))
        layers1 -=1

    model.add(layers.Flatten())

    
    layers2 = cnn_num
    while(layers2 > 0):
        keras.layers.Dropout(0.1, noise_shape=None, seed=123)
        model.add(layers.Dense(cnn_size, activation='relu', ))
        print ("cnn size - " + str(cnn_size) + "layer - " + str(layers2))
        cnn_size /= 2
        layers2-=1

    model.add(layers.Dense(2, activation=None, ))
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy', 'mse'])


    #filepath = 'weights.best.hdf5'
    #checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    #callback_list= [checkpoint]
    
    print ("training?",train)
    
    model.summary()
    model.get_config()
    model.layers[0].get_config()
    model.layers[0].input_shape			
    model.layers[0].output_shape			
    model.layers[0].get_weights()
    np.shape(model.layers[0].get_weights()[0])
    model.layers[0].trainable
    
    if train != 0 : 

        #fit the model
        print ("Using", opt, "optimizer")
        hist0 = model.fit(X_train, y_train, batch_size=256, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))
        #scores = model.evaluate(X_train, y_train, verbose=0)
        #print (scores)
        #print (model.metrics_names)
        #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")
        json_file.close()
        
    else :

        print ("check")
        
        # load json and create model
        #json_file = open('model.json', 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
        print("Loaded model from disk")

    pred_test=model.predict(X_test,batch_size=128)
    pred_train=model.predict(X_train,batch_size=128)
    res_test=np.transpose(pred_test-y_test)
    res_train=np.transpose(pred_train-y_train)
    y_test_th = np.transpose(y_test)[0]
    y_test_r = np.transpose(y_test)[1]
    y_train_th = np.transpose(y_train)[0]
    y_train_r = np.transpose(y_train)[1]
    print ("mse (theta,r) train:",split_mse(y_train,pred_train))
    print ("mse (theta,r) test:",split_mse(y_test,pred_test))     
 #   print(train_acc)
 
    train_loss = hist0.history['loss']
    train_len = len(train_loss)-1

    val_loss = hist0.history['val_loss']

    train_acc = hist0.history['acc']
    val_acc = hist0.history['val_acc']

    #print to textfile
    f.write("Optimizer:\t\t" + str(opt) +"\nPooling Dim:\t\t" +  str(Pdim)+ "\nMSE-Train(theta,r):\t" + str(split_mse(y_train,pred_train)))
    f.write("\nMSE-Test(theta,r):\t" +  str(split_mse(y_test,pred_test))+ "\nTrain Loss:\t\t" + str(train_loss[train_len]))
    f.write("\nValidation Loss:\t" + str(val_loss[train_len]) + "\nTrain_Accuracy:\t\t" + str(train_acc[train_len]) + "\nValidation Accuray:\t" +str(val_acc[train_len]))
    f.write("\n--------------------------------------------------------------------\n\n")


 
    #train_loss=hist0.history['loss']
    #print(train_loss1)
    #val_loss=hist0.history['val_loss']
    #for i in range(2):
    #    del val_loss[i]
    #print(val_loss)
    #train_acc=hist0.history['acc']
    #val_acc=hist0.history['val_acc']
    xc = range(num_epoch)
      
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
    plt.savefig("Train_loss vs Val_loss")
 
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
    plt.savefig("train_acc vs val_acc")
    
   
'''
    fig, axes = plt.subplots(4, 2)
    
    axes[0, 0].scatter(y_test_th*90.,y_test_r,alpha=0.2)
    axes[0, 0].scatter(y_train_th*90.,y_train_r,alpha=0.2)
    axes[0, 0].legend(["test","train"])
    
    axes[0, 1].scatter(res_test[0]*90.,res_test[1],alpha=0.2)
    axes[0, 1].scatter(res_train[0]*90.,res_train[1],alpha=0.2)
    #axes[0, 1].xlabel('theta residual [1/90 degrees]')
    #axes[0, 1].ylabel('radius residual [frac. of img width]')
    #axes[0, 1].legend(["test","train"])
    
    axes[1,0].hist(res_test[0]*90.,bins=np.arange(-90,90,5),histtype='step')
    axes[1,0].hist(res_train[0]*90.,bins=np.arange(-90,90,5),histtype='step')
    #axes[1,0].legend(['theta:test','theta:train'])
    
    axes[1,1].hist(res_test[1],bins=np.arange(-1,1,0.05),histtype='step')
    axes[1,1].hist(res_train[1],bins=np.arange(-1,1,0.05),histtype='step')
    #axes[1,1].legend(['r:test','r:train'])
    
    axes[2, 0].scatter(y_test_th*90.,res_test[1],alpha=0.2)
    axes[2, 0].scatter(y_train_th*90.,res_train[1],alpha=0.2)
    #axes[2, 0].ylabel('theta resid')
    #axes[2, 0].xlabel('true r')
    #axes[2, 0].legend(["test","train"])
    
    axes[2, 1].scatter(y_test_r,res_test[0]*90.,alpha=0.2)
    axes[2, 1].scatter(y_train_r,res_train[0]*90.,alpha=0.2)
    #axes[2, 1].ylabel('theta resid')
    #axes[2, 1].xlabel('true r')
    #axes[2, 1].legend(["test","train"])
    
    axes[3, 0].scatter(y_test_th*90.,res_test[0]*90.,alpha=0.2)
    axes[3, 0].scatter(y_train_th*90.,res_train[0]*90.,alpha=0.2)
    #axes[3, 0].ylabel('theta resid')
    #axes[3, 0].xlabel('true r')
    #axes[3, 0].legend(["test","train"])
    
    axes[3, 1].scatter(y_test_r,res_test[1],alpha=0.2)
    axes[3, 1].scatter(y_train_r,res_train[1],alpha=0.2)
    #axes[3, 1].ylabel('theta resid')
    #axes[3, 1].xlabel('true r')
    #axes[3, 1].legend(["test","train"])
    
    plt.show()
'''  


if __name__ == '__main__':
    imgs,labels=load_data()
    y_pred = [0,0]
    optimizer_arr = ['adam', 'rmsprop', 'nadam', 'sgd', 'adagrad', 'adadelta', 'adamax']
    f = open("Paramenter_optimization_results.txt", "a+")

    for i in optimizer_arr:
        for j in range(2,4):
            print ("\n'\noptimizer " + str(i) + " Pdim - " + str(j) + "\n\n")
            CNN(imgs,labels,y_pred, 1, i, j, j, f, 10)

    #optimizer_arr = ['adam', 'rmsprop', 'nadam', 'sgd', 'adamax', 'adadelta', 'adagrad']
    #for i in optimizer_arr:
    #    CNN(imgs,labels,y_pred, 1, i)

