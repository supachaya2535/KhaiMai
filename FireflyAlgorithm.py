#exec(open("CountNN.py").read())
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from random import randrange, uniform
from CountNN import *

from DFLSheet import *
from DFLSheet import DFLSheet
from DFLSheet import DFLSheetDataSet
import gc

def cost_CountNN(pop,X_train,Y_train):
    yY_train = copy.copy(Y_train)
    yY_train[:,0] = copy.copy(yY_train[:,0]*ratio_desired)
    Y_train_pred_proba = pop.predict(X_train,'Counting rate')     
    cost = cal_countRate(yY_train,Y_train_pred_proba)    
    return cost

def acc_CountNN(pop,X_train,Y_train):
    Y_train_pred_proba = pop.predict(X_train,'Accuracy')
    Y_train_pred = np.argmax(Y_train_pred_proba, axis=1)
    Y_train = copy.copy(np.argmax(Y_train, axis=1))
    acc = accuracy_score(Y_train,Y_train_pred)*100    
    return acc

def cal_velocityCountNN(popi,popj,nVar,beta0,gamma,alpha,norm0,norm1,delta):
    wi = copy.copy(np.array(popi['position'].countNN.layers[0].get_weights()))   
    wj = copy.copy(np.array(popj['position'].countNN.layers[0].get_weights())) 
    ui = copy.copy(np.array(popi['position'].countNN.layers[2].get_weights()))   
    uj = copy.copy(np.array(popj['position'].countNN.layers[2].get_weights())) 
    dim = (wi.shape[1]*wi.shape[2])+(ui.shape[1]*ui.shape[2])
    
    Wi = np.concatenate((wi.reshape(1,(wi.shape[1]*wi.shape[2])), ui.reshape(1,(ui.shape[1]*ui.shape[2]))),axis=1)
    Wj = np.concatenate((wj.reshape(1,(wj.shape[1]*wj.shape[2])), uj.reshape(1,(uj.shape[1]*uj.shape[2]))),axis=1)
    
    rij = np.linalg.norm(Wi-Wj) 
    beta = beta0*np.exp(-1*gamma*rij**2);
    e = delta*np.random.uniform(norm0,norm1,(1,dim))
    newWi = Wi + beta*(Wj-Wi) + alpha*e;
    wi = newWi[0][0:(wj.shape[1]*wj.shape[2])].reshape((1,wj.shape[1],wj.shape[2]))
    ui = newWi[0][(wj.shape[1]*wj.shape[2]):].reshape((1,uj.shape[1],uj.shape[2]))
    popi['position'].countNN.layers[0].set_weights(wi)  

    popi['position'].countNN.layers[2].set_weights(ui)   
    return copy.copy(popi['position'])

def write_continue_fa_file(filename,header,line): 
    try:
        f = pd.read_csv(filename).drop(['Unnamed: 0'],axis=1)
        f = f.append(line,ignore_index=True)
        f.to_csv(filename)
        print("fa continue loaded")
        
    except:
        f = pd.DataFrame(columns=header)
        f = f.append(line,ignore_index=True)
        f.to_csv(filename)
        print("create --> fa continue_"+filename)

def save_history_firefly_to_graph(history_best,dataname_fa):
    print(history_best.keys())
    fig = plt.figure()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    ax = fig.gca()
    x = range(1, len(history_best['acc'])+1)
    
    plt.plot(x,history_best['acc'])
    plt.plot(x,history_best['val_acc'])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Accuracy')
    plt.legend(['Training set','Validation set'], loc='upper left')
    plt.grid(True)
    plt.show()
 
    fig.savefig(dataname_fa +'_AccFa.png', dpi=fig.dpi)
    
    fig = plt.figure()
    plt.ylabel('counting rate')
    plt.xlabel('epoch')
    ax = fig.gca()
    plt.plot(x,history_best['counting_rate'])
    plt.plot(x,history_best['val_counting_rate'])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Counting Rate')
    plt.legend(['Training set','Validation set'], loc='upper left')
    plt.grid(True)
    plt.show()
    
    fig.savefig(dataname_fa +'_CountingRateFa.png', dpi=fig.dpi)
    
def reset_weights(model):
        session = K.get_session()
        for layer in model.layers: 
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
                
def FireFly(npop,cost_fn,lower_b ,upper_b,dimension,max_iter,fa_option,nn_option,X_train, Y_train,X_val,Y_val,X_test,Y_test,dataname_fold):
    
    empty_population = {
        'position': None,
        'cost': None,
        'best_position': None,
        'best_cost': None,
    };
            
    # Extract FireFly Params
    beta0 = fa_option['beta'] #brightness factor 0.2
    gamma = fa_option['gamma'] #light absorption coefficient () to 1 
    alpha_start = fa_option['alpha0'] #0.5
    alpha_end = fa_option['alpha1']
    norm0 = fa_option['norm0']
    norm1 = fa_option['norm1']
    alpha_damp = fa_option['alpha_damp']
    
    # Extract Problem Info
    CostFunction = cost_fn;
    VarMin = lower_b;
    VarMax = upper_b;
    nVar = dimension;
    delta = 0.1 #0.01
    # Initialize Global Best
    gbest = {'position': None, 'cost': -np.inf,'cost_val': -np.inf, 'index': 0};
    
    history_firefly_best_loss = pd.DataFrame(columns=['best_index','acc','val_acc','blind_acc','counting_rate','val_counting_rate','blind_counting_rate'])
    history_firefly_pop_loss = pd.DataFrame(columns=['pop'+ str(i) for i in range(npop)])
    # Initail population n fireflys
    pop = [];
    for i in range(0, npop):
        pop.append(empty_population.copy())
        pop[i]['position'] = copy.copy(CountNN(nn_option))
        reset_weights(pop[i]['position'].initNN)
        pop[i]['position'].initCountNN(X_train,Y_train,X_val,Y_val,
           epochs = nn_option['epochs'],batch_size=nn_option['batch_size']) #epochs =fresh:15,allblue=30,shell=70 
        
        pop[i]['cost'] = cost_CountNN(pop[i]['position'],X_train,Y_train)
        pop[i]['best_position'] = copy.copy(pop[i]['position'])
        pop[i]['best_cost'] = copy.copy(pop[i]['cost'])
        #print('Pop['+str(i)+'] is initailed')
        print('Pop[{}]:  Initail Cost = {}'.format(i,pop[i]['cost']))
        
        
        if pop[i]['best_cost'] > gbest['cost']:
            gbest['position'] = copy.copy(pop[i]['best_position'])
            gbest['cost'] = copy.copy(pop[i]['best_cost'])
            gbest['index'] = copy.copy(i)
            gbest['cost_val'] = cost_CountNN(pop[i]['position'],X_val,Y_val)
            gbest['cost_blind'] = cost_CountNN(pop[i]['position'],X_test,Y_test)
                            
    
    print('Iteration {}: Best index = {} : Best Train Cost = {}'.format(0,gbest['index'], gbest['cost']))
    # PSO Loop
    alpha = alpha_start
    for it in range(1, max_iter+1):
        for i in range(0, npop):
            pop[i]['cost'] = cost_CountNN(pop[i]['position'],X_train,Y_train)
            for j in range(0,npop):
                pop[j]['cost'] = cost_CountNN(pop[j]['position'],X_train,Y_train)
                if pop[i]['cost'] < pop[j]['cost']:
                    pop[i]['position'] = copy.copy(cal_velocityCountNN(pop[i],pop[j],nVar,beta0,gamma,alpha,norm0,norm1,delta))
                    pop[i]['cost'] = cost_CountNN(pop[i]['position'],X_train,Y_train,)
                    #print('Iter ',it,'/',max_iter,'new cost pop[',i,']:',pop[i]['cost'])
                    if pop[i]['cost'] > pop[i]['best_cost']:
                        pop[i]['best_position'] = copy.copy(pop[i]['position'])
                        pop[i]['best_cost'] = copy.copy(pop[i]['cost'])
                        
                        if pop[i]['best_cost'] > gbest['cost']:
                            #print('Iter ',it,'/',max_iter,'update the best pop[',gbest['index'],']:',gbest['cost'],' to pop[',i,'] :',pop[i]['best_cost'])
                            gbest['position'] = copy.copy(pop[i]['best_position'])
                            gbest['cost'] = copy.copy(pop[i]['best_cost'])
                            gbest['index'] = copy.copy(i)
                            gbest['cost_val'] = cost_CountNN(pop[i]['position'],X_val,Y_val)
                            gbest['cost_blind'] = cost_CountNN(pop[i]['position'],X_test,Y_test)
                            
        if alpha >= alpha_end:
            alpha *= alpha_damp
        
        
        acc_train,count_train = gbest['position'].show_report(X_train,Y_train,'Trainning',nn_option)
        acc_val,count_val = gbest['position'].show_report(X_val,Y_val,'Validation',nn_option)
        acc_blind,count_blind = gbest['position'].show_report(X_test,Y_test,'Testing',nn_option)
        print('Iteration {}: Best index = {} : Best Train Cost = {} : Best Val Cost = {}'.format(it,gbest['index'], gbest['cost'],count_val))
        
        
        L1 = history_firefly_pop_loss.columns
        L2 = [pop[i]['cost'] for i in range(npop)]
        line_pop = pd.Series(data = {k:v for k,v in zip(L1,L2)})
        line_best = pd.Series(data = {'best_index':gbest['index'],'acc':acc_train,'val_acc':acc_val,'blind_acc':acc_blind,
                                      'counting_rate':count_train,'val_counting_rate':count_val,'blind_counting_rate':count_blind})
        history_firefly_best_loss = history_firefly_best_loss.append(line_best,ignore_index=True)
        history_firefly_pop_loss = history_firefly_pop_loss.append(line_pop,ignore_index=True)
            
    print('n hidden {}: Best Train Cost = {}: Best Val Cost = {} : Best Blind Cost = {}'.format(nn_option['n_hidden'], gbest['cost'], gbest['cost_val'], gbest['cost_blind']))
    
    return gbest,history_firefly_best_loss,history_firefly_pop_loss
