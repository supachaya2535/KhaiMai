#exec(open("CountNN.py").read())
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from random import randrange, uniform
from CountNN import *
from HDF5 import *
from DFLSheet import *
from DFLSheet import DFLSheet
from DFLSheet import DFLSheetDataSet
from FireflyAlgorithm import *
import gc
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    nn_option = {'scale': 0.5,
                 'window_size': cal_window(39,0.5),
                'channels': 3,
                'n_hidden': 50,
                'display': 0,
                'kfold': 4,
                'period': 'shell'}
    
    fa_option = {'beta': 0.1,'gamma':0.01,'alpha0':0.01,'alpha1':0.0000001,
                'alpha_damp':0.5, 'norm0':0.000000001, 'norm1':1}
    
    if nn_option['period'] == 'fresh':
        nn_option['hsv'] = 1
        nn_option['nclass'] = 2
        nn_option['epochs'] = 90
        nn_option['batch_size'] = 2
        nn_option['list_dfl'] = ['F12','F16',
                                 'F4','F5','F6','F13','F14','F15']#'F1',
        nn_option['path_dfl'] = 'F\\'
        nn_option['best_thr'] = [0.,0.5]
        
    elif nn_option['period'] == 'allblue':
        nn_option['hsv'] = 1
        nn_option['nclass'] = 3
        nn_option['epochs'] = 90
        nn_option['batch_size'] = 3
        nn_option['list_dfl'] = ['A4','F7s','F10s',
                                 'F3p','F8s','F9s','F11s']
        nn_option['path_dfl'] = 'A\\'
        nn_option['best_thr'] = [0.,0.5,0.5]
        
    elif nn_option['period'] == 'shell':
        nn_option['hsv'] = 0
        nn_option['nclass'] = 4
        nn_option['epochs'] = 150
        nn_option['batch_size'] = 3
        """
        nn_option['list_dfl'] = ['S2','S5','S6','S8','S9','S10','S11p','S12',
                                 'S13p','S14a','S15a','S16a','S17a','S18a',
                                 'S19a','S20a','S21a','S22a','S24a','S25a',
                                 'S26a','S27a','S28a','S29a','S30a','S31p','S35s']
        """
        nn_option['list_dfl'] = ['CM_S1','SB_S1','SB_S2','SB_S3','SB_S4']
        nn_option['path_dfl'] = 'S\\'
        nn_option['best_thr'] = [0.,0.5,0.5,0.5]
    
    nn_option['step_size'] = 7
    dimension = nn_option['window_size']*nn_option['window_size']*nn_option['channels']
    lower_b=-np.inf
    upper_b= np.inf
    max_iter = 15
    npop = 4
    
    ############################################################################
    input_dim = nn_option['window_size']*nn_option['window_size']*3
    nn_option['dataname'] = 'DataSet_'+nn_option['period']+'_ws'+str(nn_option['window_size'])+'_scale'+str(int(nn_option['scale']*10))+'_'+str(nn_option['kfold'])+'fold'

    X, Y, xX_test, xy_test = load_signature_lib(nn_option['dataname'],nn_option)
    
    header = ['nhidden','f1','f2','f3','f4']
    
    for nh in range(50,51,5):  #25-55 
        count_train_fa = pd.Series()
        count_val_fa = pd.Series()
        count_blind_fa = pd.Series()
        acc_train_fa = pd.Series()
        acc_val_fa = pd.Series()
        acc_blind_fa = pd.Series()
        count_val_fa_gbest = pd.Series()
        
        count_train_fa['nhidden'] = nh
        count_val_fa['nhidden'] = nh
        count_blind_fa['nhidden'] = nh
        acc_train_fa['nhidden'] = nh
        acc_val_fa['nhidden'] = nh
        acc_blind_fa['nhidden'] = nh
        count_val_fa_gbest['nhidden'] = nh
        
        nn_option['n_hidden'] = nh
        
        if os.path.exists(os.path.dirname(nn_option['dataname']+'/countnn_fa'+str(nn_option['n_hidden'])+'/')):
            #shutil.rmtree(os.path.dirname(nn_option['dataname']+'/countnn_fa'+str(nn_option['n_hidden'])+'/'))
            print(str(nn_option['n_hidden']) + ' hidden nodes is exist')
        else:
            os.makedirs(os.path.dirname(nn_option['dataname']+'/countnn_fa'+str(nn_option['n_hidden'])+'/'))
            os.makedirs(os.path.dirname(nn_option['dataname']+'/countnn_fa'+str(nn_option['n_hidden'])+'/Result_img/'))
            os.makedirs(os.path.dirname(nn_option['dataname']+'/countnn_fa'+str(nn_option['n_hidden'])+'/Result_MSP_img/'))
        
        dataname_fa = nn_option['dataname']+'/countnn_fa'+str(nn_option['n_hidden'])+'/'
        nn_option['dataname_fa'] = dataname_fa
        
        for f in range(1,nn_option['kfold']+1-3):
            dataname_fold = nn_option['dataname']+"/fold"+str(f)
            
            X_train, y_train,X_val, y_val = load_signature_lib_cv(dataname_fold,f)
            X_train = copy.copy(transform_img2vec(X_train, input_dim))
            X_train = copy.copy(X_normalize(X_train))
            scaler = preprocessing.StandardScaler().fit(X_train)
            #
            #X_train = scaler.transform(X_train).copy()
            Y_train = copy.copy(np_utils.to_categorical(y_train,nn_option['nclass']))
            X_test = copy.copy(transform_img2vec(xX_test, input_dim))
            X_test = copy.copy(X_normalize(X_test))
            #
            #X_test = scaler.transform(X_test).copy()
            Y_test = copy.copy(np_utils.to_categorical(xy_test,nn_option['nclass']))
            X_val = copy.copy(transform_img2vec(X_val, input_dim))
            X_val = copy.copy(X_normalize(X_val))
            #
            #X_val = scaler.transform(X_val).copy()
            Y_val = copy.copy(np_utils.to_categorical(y_val,nn_option['nclass']))

            countnn,history_firefly_best_loss,history_firefly_pop_loss = FireFly(npop,cost_CountNN,lower_b,upper_b,dimension,max_iter,fa_option,
                                                                               nn_option,X_train, Y_train,X_val,Y_val,X_test,Y_test,dataname_fold)
    
            count_val_fa_gbest['f'+str(f)] = countnn['cost_val']
            
            #### Save model
            countnn['position'].countNN.save_weights(dataname_fa +'nh'+str(nn_option['n_hidden'])+'_fold_'+str(f)+'_mdlf.hdf5')
            print('Fold:'+str(f)+'/'+str(nn_option['kfold'])+' Saved model hidden:'+str(nn_option['n_hidden'])+' to disk')
            
            #### Save result
            history_firefly_best_loss.to_csv(dataname_fa +'history_firefly_best_loss_fold_'+str(f)+'.csv')
            history_firefly_pop_loss.to_csv(dataname_fa +'history_firefly_pop_loss_fold_'+str(f)+'.csv')
            save_history_firefly_to_graph(history_firefly_best_loss,dataname_fa+'fold_'+str(f))
            
            #visualize_weight(countnn, 20)
            
            countnn['position'].countNN.load_weights(dataname_fa+'nh'+str(nn_option['n_hidden'])+'_fold_'+str(f)+'_mdlf.hdf5')
            
            nn_option['display'] = 1
            countnn['position'].show_report(X_test,Y_test,'Testing',nn_option)
            nn_option['display'] = 0
            
            count_train_fa['f'+str(f)] = cost_CountNN(countnn['position'],X_train,Y_train)
            count_val_fa['f'+str(f)] = cost_CountNN(countnn['position'],X_val,Y_val)
            count_blind_fa['f'+str(f)] = cost_CountNN(countnn['position'],X_test, Y_test)
            acc_train_fa['f'+str(f)] = acc_CountNN(countnn['position'],X_train,Y_train)
            acc_val_fa['f'+str(f)] = acc_CountNN(countnn['position'],X_val,Y_val)
            acc_blind_fa['f'+str(f)] = acc_CountNN(countnn['position'],X_test, Y_test)
            
            gc.collect()
            
            nn_option['display'] = 1
            #for dfl in list_dfl:
               #countnn['position'].predict_class(path_dfl,dfl,scaler,nn_option)
            nn_option['display'] = 0
            #if f ==1 :
                #countR = test_on_DFL_Train(scaler,nn_option,countnn)
            
            
        write_continue_fa_file(nn_option['dataname'] +'/result_count_train_fa.csv',header,count_train_fa)
        write_continue_fa_file(nn_option['dataname'] +'/result_count_val_fa.csv',header,count_val_fa)
        write_continue_fa_file(nn_option['dataname'] +'/result_count_blind_fa.csv',header,count_blind_fa)
        write_continue_fa_file(nn_option['dataname'] +'/result_acc_train_fa.csv',header,acc_train_fa)
        write_continue_fa_file(nn_option['dataname'] +'/result_acc_val_fa.csv',header,acc_val_fa)
        write_continue_fa_file(nn_option['dataname'] +'/result_acc_blind_fa.csv',header,acc_blind_fa)
        write_continue_fa_file(nn_option['dataname'] +'/result_count_val_fa_gbest.csv',header,count_val_fa_gbest)
    
    #loaded_model.load_weights("model.h5")
    #print("Loaded model from disk")
    ############################################################################
    