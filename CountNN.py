from DFLSheet import DFLSheet,DFLSheetDataSet
from HDF5 import *

import matplotlib.pyplot as plt
from PIL import Image
import copy
import os
import random
import shutil
import numpy as np
import pandas as pd
from keras import optimizers
from keras import backend as K
from keras.datasets import *
from keras.layers import * 
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import *
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from skimage import io
from skimage.draw import circle
from skimage.transform import rescale, resize  
from sklearn.metrics import *
from sklearn import preprocessing

ratio_desired = 1
class CountNN():
    def __init__(self,nn_option):
        random.seed(9001)
        self.img_rows = nn_option['window_size']
        self.img_cols = nn_option['window_size']
        self.img_channels = nn_option['channels']
        self.n_classes = nn_option['nclass']
        self.nuerons = nn_option['n_hidden']
        self.input_dim = int(self.img_rows*self.img_cols*self.img_channels)
        self.initNN = copy.copy(self.build_count_nn())
        
        #self.initNN.summary()
        self.classNN = copy.copy(self.initNN)
        self.countNN = copy.copy(self.initNN)
        self.dataname = nn_option['dataname']
        
    def build_count_nn(self):
        model = Sequential()
        model.add(Dense(self.nuerons,input_shape=(self.input_dim,),use_bias=False))
        model.add(Activation('sigmoid'))
        model.add(Dense(self.n_classes, activation='sigmoid',use_bias=False))
        return copy.copy(model)
    
    
    
    def initCountNN(self,X_train, Y_train, X_test, Y_test,epochs,batch_size):
        yY_train = copy.copy(Y_train)
        yY_train[:,0] = yY_train[:,0]*ratio_desired
        yY_test = copy.copy(Y_test)
        yY_test[:,0] = yY_test[:,0]*ratio_desired
        sgd = optimizers.SGD(lr=0.01, decay=1e-13, momentum=0.9, nesterov=True)
        self.initNN.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
        self.initNN.fit(X_train, yY_train,validation_data=(X_test, yY_test),epochs=epochs, 
                                    batch_size= batch_size,verbose = 0)
        self.countNN.set_weights(self.initNN.get_weights().copy())
        self.classNN.set_weights(self.countNN.get_weights().copy())
       
    
    def trainCountNN(self,X_train, y_train, X_test, y_test,epochs,batch_size):
        yY_train = copy.copy(y_train)
        yY_train[:,0] = yY_train[:,0]*ratio_desired
        yY_test = copy.copy(y_test)
        yY_test[:,0] = yY_test[:,0]*ratio_desired
        
        history_init_loss = pd.DataFrame()
        history_class_loss = pd.DataFrame()
        history_count_loss = pd.DataFrame()
        
        if os.path.exists(os.path.dirname(self.dataname+'/initnn/')):
            shutil.rmtree(os.path.dirname(self.dataname+'/initnn/'))
        os.makedirs(os.path.dirname(self.dataname+'/initnn/'))
            
        if os.path.exists(os.path.dirname(self.dataname+'/classnn/')):
            shutil.rmtree(os.path.dirname(self.dataname+'/classnn/'))
        os.makedirs(os.path.dirname(self.dataname+'/classnn/'))
            
        if os.path.exists(os.path.dirname(self.dataname+'/countnn/')):
            shutil.rmtree(os.path.dirname(self.dataname+'/countnn/'))
        os.makedirs(os.path.dirname(self.dataname+'/countnn/'))
        
        cp_init = ModelCheckpoint(self.dataname+'/initnn/{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        cp_class = ModelCheckpoint(self.dataname+'/classnn/best_mdlf.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        cp_count = ModelCheckpoint(self.dataname+'/countnn/best_mdlf.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        
        sgd1 = optimizers.SGD(lr=0.0001, decay=1e-13, momentum=0.9, nesterov=True)
        sgd2 = optimizers.SGD(lr=1e-11, decay=1e-13, momentum=0.9, nesterov=True)
        self.initNN.compile(loss='mean_squared_error', optimizer= sgd1, metrics=['accuracy'])
        self.classNN.compile(loss='mean_squared_error', optimizer= sgd1, metrics=['accuracy'])
        self.countNN.compile(loss = counting_loss, optimizer=sgd2, metrics=['accuracy',counting_rate])
            
        for epoch in range(epochs):
            print( '========== Train InitNN ==========')
            init_loss = self.initNN.fit(X_train, yY_train,validation_data=(X_test, yY_test),epochs=epochs, batch_size = batch_size,verbose = 0)
            history_init_loss = pd.concat([history_init_loss,pd.DataFrame(init_loss.history)],ignore_index=True)
            print(init_loss.history.keys())
            
            print( '========== Train ClassNN ==========')
            self.classNN.set_weights(self.initNN.get_weights()) 
            class_loss = self.classNN.fit(X_train, yY_train,validation_data=(X_test, yY_test) , epochs=epochs, batch_size= batch_size, callbacks=[cp_class],verbose = 0)
            history_class_loss = pd.concat([history_class_loss,pd.DataFrame(class_loss.history)],ignore_index=True)
            
            print('========== Train CountNN ===========')
            self.countNN.set_weights(self.initNN.get_weights())   
            count_loss = self.countNN.fit(X_train, yY_train,validation_data=(X_test, yY_test) , epochs=epochs, batch_size=X_train.shape[0], callbacks=[cp_count],verbose = 0)
            history_count_loss = pd.concat([history_count_loss,pd.DataFrame(count_loss.history)],ignore_index=True)
            
            # Plot the progress
            #print ("%d [ClassNN loss: %f, acc.: %.2f%%] [CountNN loss: %f, acc.: %.2f%%]" % (epoch, class_loss.history['loss'][len(class_loss.history['loss'])-1], class_loss.history['acc'][len(class_loss.history['acc'])-1],count_loss.history['loss'][len(count_loss.history['loss'])-1], count_loss.history['acc'][len(count_loss.history['acc'])-1]))
        
        return history_init_loss,history_class_loss,history_count_loss
    
    def predict(self,X,params):
        Y = self.countNN.predict(X)
        Y_pred = np.argmax(Y, axis=1)
        Y_pred_code = copy.copy(np_utils.to_categorical(Y_pred,Y.shape[1]))
        return Y_pred_code
    
    def predict_thr(self,X,thr):
        Y = self.countNN.predict(X)
        Y_pred = np.argmax(Y, axis=1)
        #Y[Y[:,0]>0,0] = 0
        Y_pred_code = copy.copy(np_utils.to_categorical(Y_pred,Y.shape[1]))
        #Y[Y_pred==0,0] = 1   
        Y = Y*Y_pred_code
        for i in range(1,len(thr)):
            Y[Y[:,i]<thr[i],i] = 0    
        return Y
        
    def predict_class(self,rootpath,dflsheet_name,scaler,params):
        dfl = DFLSheet(rootpath,dflsheet_name,params['period'],params['scale'],params['nclass'],params['hsv'])
        print(dflsheet_name+' loading was finished')
        dfl.crop_egg(params['window_size'])
        #dfl.crop_egg_all(params['window_size'])
        X = copy.copy(np.array(dfl.eggImg))
        X = copy.copy(transform_img2vec(X, self.input_dim))
        X = copy.copy(X_normalize(X))
        #
        #X = scaler.transform(X).copy()
        Y = copy.copy(np_utils.to_categorical(dfl.eggLabel,params['nclass']))
        acc_train,count_train = self.show_report(X,Y,dflsheet_name,params)
    
    def predict_dfl(self,dflsheet_name,scaler,params):
        rootpath = params['path_dfl']
        step_size = params['step_size']
        thr = params['best_thr']
        dfl = DFLSheet(rootpath,dflsheet_name,params['period'],params['scale'],params['nclass'],params['hsv'])
        print(dflsheet_name+' loading in predict_dfl was finished')
        dfl.crop_all(params['window_size'],step_size)
        
        X = copy.copy(np.array(dfl.allImg))
        eggPoint = np.array(dfl.allPoint)
        X = copy.copy(transform_img2vec(X, self.input_dim))
        X = copy.copy(X_normalize(X))
        #
        #X = scaler.transform(X).copy()
        Y = self.predict_thr(X,thr)

        if True:
            rgb = dfl.OrigImgLim
            draw_roi_egg(Y,rgb,eggPoint,params,dflsheet_name,dfl.egg_num_lim)
            
        y_pred_eggnum = Y.sum(axis=0)
        num_desired = pd.DataFrame(data = [dfl.egg_num_lim], columns = dfl.egg_type, index = [dflsheet_name])
        num_predict = pd.DataFrame(data = [y_pred_eggnum.astype('float32')], columns = dfl.egg_type, index = [dflsheet_name])
        return num_desired,num_predict 
    
    def predict_dfl_NonMaximunSup(self,dflsheet_name,scaler,params):
        rootpath = params['path_dfl']
        step_size = params['step_size']
        thr = params['best_thr']
        dfl = DFLSheet(rootpath,dflsheet_name,params['period'],params['scale'],params['nclass'],params['hsv'])
        print(dflsheet_name+' loading in predict_dfl was finished')
        dfl.crop_all(params['window_size'],step_size)
        
        X = copy.copy(np.array(dfl.allImg))
        eggPoint = np.array(dfl.allPoint)
        X = copy.copy(transform_img2vec(X, self.input_dim))
        X = copy.copy(X_normalize(X))
        #X = scaler.transform(X).copy()
        Y = self.predict_thr(X,thr)

        if True:
            rgb = dfl.OrigImgLim
            y_pred_eggnum,count_score = draw_NMS_egg(Y,rgb,eggPoint,params,dflsheet_name,dfl.egg_num_lim)
            
        #y_pred_eggnum = Y.sum(axis=0)
        num_desired = pd.DataFrame(data = [dfl.egg_num_lim], columns = dfl.egg_type, index = [dflsheet_name])
        num_predict = pd.DataFrame(data = [y_pred_eggnum.astype('float32')], columns = dfl.egg_type, index = [dflsheet_name])
        print(num_desired)
        print(num_predict)
        print(count_score)
        return num_desired,num_predict,count_score 
    
    def show_report(self,X,Y,name_data,params):
        Y_pred = self.predict(X,params)
        Y_pred_ = np.argmax(Y_pred, axis=1)
        Y_ = np.argmax(Y, axis=1)
        conf_matrix = confusion_matrix(Y_, Y_pred_) 
        acc_score =  accuracy_score(Y_, Y_pred_)*100 
        count_score = cal_countRate(Y, Y_pred) 
        
        classwise = cal_classwise( conf_matrix)
        
        if params['display']:
            print('========CountNN  Report========')
            print(name_data," %s Score: %.4f%%" % ('classification rate', acc_score))     
            print(name_data," %s Score: %.4f%%" % ('counting rate', count_score ))    
            print(name_data," class-wise: ", classwise )     
            print(name_data," : Confusion Matrix: \n", conf_matrix) 
            print('Desired :',np.sum(Y,0))
            print('Predict :',np.sum(Y_pred,0))
        return acc_score,count_score

    def select_stebsize(self,rootpath,dflsheet_name,step_size,scaler, nn_option):
        nn_option['display'] = 0
        dflsheet_data = DFLSheetDataSet(nn_option)
        dflsheet_data.create_all_subimg_dfl(step_size,dflsheet_name)
        dflsheet_data.egg_num.to_csv(nn_option['dataname'] +'/'+nn_option['period']+'EggNumber.csv')
        for s in range(6,11):
            step_result =  []
            for i in range(len(dflsheet_name)):
                 num_desired ,num_predict = myNN.predict_dfl(rootpath,dflsheetlist[i],step_size,scaler,nn_option)
                 line_desire = pd.DataFrame([num_desired], columns = dfl.egg_type, index = [dflsheetlist[i]])
                 line_predict = pd.DataFrame([num_predict], columns = dfl.egg_type, index = [dflsheetlist[i]])
            
        print(num_predict)
        print(num_desired)
