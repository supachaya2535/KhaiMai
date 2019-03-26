import numpy as np
from keras import backend as K
import h5py
import matplotlib.pyplot as plt
import os
import math
import errno
import pandas as pd
import copy
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from DFLSheet import *
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow

def save_img_as_HDF5(hdf5_path,img_train,label_train):
    img = img_train[0];
    if data_order == 'th':
        train_shape = (len(label_train), img.shape[2], img.shape[0], img.shape[1])
    elif data_order == 'tf':
        train_shape = (len(label_train), img.shape[0], img.shape[1], img.shape[2])
        
    hdf5_file = h5py.File(hdf5_path, mode='w')
    hdf5_file.create_dataset("images", train_shape, np.float16)
    hdf5_file.create_dataset("labels", (len(label_train),), np.float16)
    hdf5_file["labels"][...] = label_train
   
    # loop over train addresses
    for i in range(len(label_train)):
        # print how many images are saved every 1000 images
        if i % 10000 == 0 and i > 1:
            print('Image data: {}/{}'.format(i, len(label_train)))
        # read an image and resize to (224, 224)
        img = img_train[i]
        # if the data order is Theano, axis orders should change
        if data_order == 'th':
            img = np.rollaxis(img, 2)
        # save the image and calculate the mean so far
        hdf5_file["images"][i, ...] = img
    # save the mean and close the hdf5 file
    hdf5_file.close()
    
def load_hdf5_as_image(hdf5_path):
    hdf5_file = h5py.File(hdf5_path, "r")
    # Total number of samples
    data_num = hdf5_file["images"].shape[0]
    datas = hdf5_file["images"].value
    labels = hdf5_file["labels"].value
    hdf5_file.close()
    
    return datas, labels

def splite_dataset_cv(kfold,DataName,X,y):
    # Instantiate the cross validator
    skf = list(StratifiedKFold(n_splits=kfold, shuffle=True).split(X, y))
    # Loop through the indices the split() method returns
    SigEggNumber_fold = pd.DataFrame()
    for index, (train_idx, val_idx) in enumerate(skf):
        
        print("Save fold# " ,index+1,"/",kfold,"...")
        
        # Generate batches from indices
        xtrain, xval = X[train_idx], X[val_idx]
        ytrain, yval = y[train_idx], y[val_idx] 
        
        if not os.path.exists(os.path.dirname(DataName+"/fold"+str(index+1)+'/')):
            try:
                os.makedirs(os.path.dirname(DataName+"/fold"+str(index+1)+'/'))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        hdf5_train = DataName+"/fold"+str(index+1)+"/train.hdf5"
        hdf5_val = DataName+"/fold"+str(index+1) +"/val.hdf5"
    
        save_img_as_HDF5(hdf5_train, xtrain, ytrain)
        save_img_as_HDF5(hdf5_val, xval, yval)
        u, indices = np.unique(yval, return_counts=True)
        egg_num = indices
    
        line_num = pd.DataFrame([egg_num], columns = u, index = ['fold'+str(index)])
        SigEggNumber_fold = SigEggNumber_fold.append(line_num)
    SigEggNumber_fold.to_csv(DataName +'/SigEggNumber_fold.csv')
           
############################################################################## 
def counting_loss(y_true,y_pred):
    sum_y_true = copy.copy(K.sum(y_true, axis=0))
    sum_y_pred = copy.copy(K.sum(y_pred, axis=0))
    return ((K.sum(K.abs(sum_y_true-sum_y_pred), axis=1))/K.sum(K.abs(sum_y_true), axis=1))*100

def counting_rate(y_true,y_pred):
    sum_y_true = copy.copy(K.sum(y_true, axis=0))
    sum_y_pred = copy.copy(K.sum(y_pred, axis=0))
    return (1-((K.sum(K.abs(sum_y_true-sum_y_pred), axis=1))/K.sum(K.abs(sum_y_true), axis=1)))*100

def cal_countRate(y_true,y_pred):
    sum_y_true = copy.copy(np.sum(y_true, axis=0))
    sum_y_pred = copy.copy(np.sum(y_pred, axis=0))
    return (1-(np.sum(np.abs(sum_y_true-sum_y_pred))/np.sum(np.abs(sum_y_true))))*100

def cal_sum_countRate(sum_y_true,sum_y_pred):
    return (1-np.sum(np.abs(sum_y_true-sum_y_pred))/np.sum(np.abs(sum_y_true)))*100

def cal_countLoss(y_true,y_pred):
    sum_y_true =  copy.copy(np.sum(y_true, axis=0).astype('float16'))
    sum_y_pred =  copy.copy(np.sum(y_pred, axis=0).astype('float16'))
    return (np.sum(np.abs(sum_y_true-sum_y_pred))/np.sum(np.abs(sum_y_true)))*100

def cal_countRate_per_class(y_true,y_pred):
    sum_y_true = copy.copy(np.sum(y_true, axis=0).astype('float16'))
    sum_y_pred = copy.copy(np.sum(y_pred, axis=0).astype('float16'))
    return (1-(np.abs(sum_y_true-sum_y_pred)/np.abs(sum_y_true)))*100

def cal_classwise(conf_matrix):
    sum_y_pred = np.zeros(conf_matrix.shape[0])
    sum_y_true = np.zeros(conf_matrix.shape[0])
    for i in range(0,conf_matrix.shape[0]):
        sum_y_pred[i] = conf_matrix[i][i]
        sum_y_true[i] = np.sum(conf_matrix[i][:])
    #print(sum_y_pred , sum_y_true)
    return (1-((np.abs(sum_y_true-sum_y_pred))/sum_y_true))*100

def save_history_to_graph(history_init,history_class,history_count, path):
    print(history_class.keys())
    fig = plt.figure()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    #plt.plot(history_init['acc'])
    plt.plot(history_class['acc'])
    plt.plot(history_count['acc'])
    plt.title('Accuracy on Training set')
    plt.legend(['classNN','countNN'], loc='upper left')
    plt.show()
    fig.savefig(path+'/AccTrain.png', dpi=fig.dpi)
    
    fig = plt.figure()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    #plt.plot(history_init['val_acc'])
    plt.plot(history_class['val_acc'])
    plt.plot(history_count['val_acc'])
    plt.title('Accuracy on Validation set')
    plt.legend(['classNN','countNN'], loc='upper left')
    plt.show()
    fig.savefig(path+'/AccVal.png', dpi=fig.dpi)
    
    fig = plt.figure()
    plt.ylabel('counting_rate')
    plt.xlabel('epoch')
    #plt.plot(history_init['val_acc'])
    plt.plot(history_class['counting_rate'])
    plt.plot(history_count['counting_rate'])
    plt.title('Counting rate on Trainning set')
    plt.legend(['classNN','countNN'], loc='upper left')
    plt.show()
    fig.savefig(path+'/counting_rate.png', dpi=fig.dpi)
    
    fig = plt.figure()
    plt.ylabel('counting_rate')
    plt.xlabel('epoch')
    #plt.plot(history_init['val_acc'])
    plt.plot(history_class['val_counting_rate'])
    plt.plot(history_count['val_counting_rate'])
    plt.title('Counting rate on Validation set')
    plt.legend(['classNN','countNN'], loc='upper left')
    plt.show()
    fig.savefig(path+'/counting_rate_val.png', dpi=fig.dpi)
    

def load_signature_lib(DataName,params):
    #DataName = params['dataname']
    kfold = params['kfold']
    hdf5_path_train = DataName +"/DFL_Data_train.hdf5"
    hdf5_path_blind = DataName +"/DFL_Data_blind.hdf5"
    
    if not os.path.exists(os.path.dirname(DataName+"/")):
        try:
            os.makedirs(os.path.dirname(DataName+"/"))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    if os.path.exists(hdf5_path_train) and (os.path.exists(hdf5_path_blind)):
        egg_num = pd.read_csv(DataName +'/EggNumber.csv')
        print(egg_num)
        sig_egg_num = pd.read_csv(DataName +'/SigEggNumber.csv')
        print(sig_egg_num)
        
        train_Img, train_Label = load_hdf5_as_image(hdf5_path_train)
        blind_Img, blind_Label = load_hdf5_as_image(hdf5_path_blind)
        print("A Signature Library for ",params['period']," period already exist")
        print("Dowloading completed....")
        
    else:
        print("A Signature Library for ",params['period']," period isn't exist. Trying to create a new one......")
        Dataset = DFLSheetDataSet(params)
        Dataset.generate_signatureLibraly()
        
        Dataset.sig_egg_num.to_csv(DataName +'/SigEggNumber.csv')
        Dataset.egg_num.to_csv(DataName +'/EggNumber.csv')
        splite_dataset_cv(kfold,DataName,Dataset.train_sigImg, Dataset.train_sigLabel)
        save_img_as_HDF5(hdf5_path_train,Dataset.train_sigImg, Dataset.train_sigLabel)
        save_img_as_HDF5(hdf5_path_blind,Dataset.blind_sigImg, Dataset.blind_sigLabel)
        #Dataset.sig_egg_num.to_csv(DataName +'/SigEggNumber.csv')
        train_Img, train_Label = load_hdf5_as_image(hdf5_path_train)
        blind_Img, blind_Label = load_hdf5_as_image(hdf5_path_blind)
        egg_num = pd.read_csv(DataName +'/EggNumber.csv')
        print(egg_num)
        sig_egg_num = pd.read_csv(DataName +'/SigEggNumber.csv')
        print(sig_egg_num)
        print("Create A New Signature Library for ",params['period']," period")
    return train_Img, train_Label,blind_Img, blind_Label

def load_signature_lib_cv(DataName,kfold):
    hdf5_train = DataName +"/train.hdf5"
    hdf5_val = DataName +"/val.hdf5"
    X_train, y_train = load_hdf5_as_image(hdf5_train)
    X_val, y_val = load_hdf5_as_image(hdf5_val)
    return X_train, y_train,X_val, y_val

def cal_window(window_size,scale):
    new_win = copy.copy(int(math.floor(window_size*scale)+(math.floor(window_size*scale)-1)%2))
    return new_win

def transform_img2vec(X,input_dim):
    X_train = X.reshape(X.shape[0],input_dim).astype('float16')
    return X_train

def draw_roi_egg(Y,rgb,eggPoint,params,name,egg_num_desired):
    from skimage.draw import circle
    from skimage import io
    nclass = params['nclass']
    Y_train_pred = np.argmax(Y, axis=1)
    confi_map = np.zeros((rgb.shape[0],rgb.shape[1],nclass+1))
    pix_colur = [ [255,0,0], [255,255,0] ,[255,0,255]]
    predict_dfl = np.zeros(nclass)
    for i in range(0,len(eggPoint)):
        xx = eggPoint[i][0]
        yy = eggPoint[i][1]
        dim = Y_train_pred[i]
        confi_map[xx][yy][dim] = Y[i][dim]
        if dim > 0:
            rr, cc = circle(xx, yy, 5, rgb.shape)
            rgb[rr, cc, :] = pix_colur[int(dim)-1]
            predict_dfl[int(dim)] = predict_dfl[int(dim)]+1
   
    font = ImageFont.truetype('Roboto-Bold.ttf', size=21)
    # Opening the file gg.png
    img1 = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img1)
    (x, y) = (5, 5)
    message = 'Desired :'+str(egg_num_desired)
    color = 'rgb(0, 0, 0)' # black color
    draw.text((x, y), message, fill=color, font=font)
    
    (x, y) = (5, 45)
    message = 'Predicted : '+ str(predict_dfl)
    color = 'rgb(0, 0, 0)' # black color
    draw.text((x, y), message, fill=color, font=font)
    
    egg_num_desired[0] = 0
    count_score = cal_sum_countRate(egg_num_desired, predict_dfl) 
    (x, y) = (5, 85)
    message = str(count_score)
    color = 'rgb(0, 0, 0)' # black color
    draw.text((x, y), message, fill=color, font=font)
    
    img1.save(params['dataname_fa'] + 'Result_img/'+name+'.png')
    print('Save DFL counting Image of '+name+' to drive')
            
def draw_NMS_egg(Y,rgb,eggPoint,params,name,egg_num_desired):
    from skimage.draw import circle
    from skimage import io
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage import data, img_as_float
    nclass = params['nclass'] 
    Y_train_pred = np.argmax(Y, axis=1)
    confi_map = np.zeros((rgb.shape[0],rgb.shape[1]))
    class_map = np.zeros((rgb.shape[0],rgb.shape[1]))
    
    
    for i in range(1,len(eggPoint)):
        xx = eggPoint[i][0]
        yy = eggPoint[i][1]
        dim = Y_train_pred[i]
        if dim>0:
            confi_map[xx][yy] = Y[i][dim]
            class_map[xx][yy] = dim
    
    #io.imsave('Result_img\\confi_' +name +'.jpg',SubImg_normalize(confi_map.copy()))
    #confi_map = ndi.maximum_filter(confi_map, size= 7, mode='constant')
    coordinates = peak_local_max(confi_map, min_distance = params['step_size']) #,threshold_abs = 0.5)
           
    pix_colur = [ [255,0,0], [255,255,0] ,[255,0,255]]
    predict_dfl = np.zeros(nclass)
    for i in range(0,len(coordinates)):
        xx = coordinates[i][0]
        yy = coordinates[i][1]
        dim = class_map[xx][yy]
        if dim > 0:
            rr, cc = circle(xx, yy, 5, rgb.shape)
            rgb[rr, cc,:] =  pix_colur[int(dim)-1]
            predict_dfl[int(dim)] = predict_dfl[int(dim)]+1
   
    font = ImageFont.truetype('Roboto-Bold.ttf', size=21)
    # Opening the file gg.png
    img1 = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img1)
    (x, y) = (5, 5)
    message = 'Desired :'+str(egg_num_desired)
    color = 'rgb(0, 0, 0)' # black color
    draw.text((x, y), message, fill=color, font=font)
    
    (x, y) = (5, 45)
    message = 'Predicted : '+ str(predict_dfl)
    color = 'rgb(0, 0, 0)' # black color
    draw.text((x, y), message, fill=color, font=font)
    
    egg_num_desired[0] = 0
    count_score = cal_sum_countRate(egg_num_desired, predict_dfl) 
    (x, y) = (5, 85)
    message = str(count_score)
    color = 'rgb(0, 0, 0)' # black color
    draw.text((x, y), message, fill=color, font=font)
    
    img1.save(params['dataname_fa'] + 'Result_MSP_img/pil'+name+'.png')
    #io.imsave('Result_NMS_img/'+name+'.png',rgb)
    print('Save NMS DFL counting Image of '+name+' to drive')
    return predict_dfl,count_score

def test_on_DFL_Test(scaler,nn_option,countnn):
    countR = {}
    desiredC = pd.DataFrame()
    predictC = pd.DataFrame()
    for dfl in nn_option['list_dfl']:
        countnn['position'].predict_dfl(dfl,scaler,nn_option)
        a,b,c = countnn['position'].predict_dfl_NonMaximunSup(dfl,scaler,nn_option)
        b['CountR'] = c
        desiredC = desiredC.append(a,ignore_index=False)
        predictC = predictC.append(b,ignore_index=False)
        countR[dfl] = c
    #print(countR)
    #predictC = pd.concat([predictC,countR], axis=1, sort=False)
    return desiredC,predictC,countR

def test_on_DFL_Train(scaler,nn_option,countnn):
    countR = {}
    desiredC = pd.DataFrame()
    predictC = pd.DataFrame()
    for dfl in nn_option['list_train']:
        countnn['position'].predict_dfl(dfl,scaler,nn_option)
        a,b,c = countnn['position'].predict_dfl_NonMaximunSup(dfl,scaler,nn_option)
        b['CountR'] = c
        desiredC = desiredC.append(a,ignore_index=False)
        predictC = predictC.append(b,ignore_index=False)
        countR[dfl] = c
    #print(countR)
    #predictC = pd.concat([predictC,countR], axis=1, sort=False)
    return desiredC,predictC,countR
    
    
    