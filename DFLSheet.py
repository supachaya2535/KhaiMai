import pandas as pd
import math
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import copy
from skimage import io
from skimage import color
from skimage.transform import rescale, resize  
from sklearn.model_selection import StratifiedKFold
import cv2
from scipy import ndimage as ndi
from PIL import ImageEnhance,Image
from matplotlib import cm
rootpath = 'E:\\OneDrive - Chiang Mai University\\CoLab_KhaiMai\\TrainImg\\'
#rootpath = 'TrainImg\\'

def X_normalize(image):
    image = image.astype('float')
    # Normalize images for better comparison.
    #image = (image - image.mean()) / image.std()
    #return np.sqrt(np.real(image)**2 +
      #             np.imag(image)**2)
    
    X_norm = image.astype('float16')/255.0
    return X_norm

def SubImg_normalize(X):
    X_norm = X.astype('float16')
    max_pix = np.max(X_norm)
    min_pix = np.min(X_norm)
    X_norm = ((X_norm - min_pix)/(max_pix-min_pix))
    X_norm = (255.0*X_norm).astype('uint8')
    return X_norm

def SubImg_normalize2(X,Min,Max):
    X_norm = X.astype('float16')
    X_norm = ((X_norm - Min)/(Max-Min))
    X_norm = (255.0*X_norm).astype('uint8')
    return X_norm

def Img_normalize(X):
    X_norm = X.astype('float16')
    max_pix = np.max(X_norm)
    min_pix = np.min(X_norm)
    X_norm = ((X_norm - min_pix)/(max_pix-min_pix))
    X_norm = (255.0*X_norm).astype('uint8')
    return X_norm

def Img_normalize_hsv(X):
    X_norm = X.astype('float16')
    #X_norm[0] = X_norm[0]*360
    #X_norm[1] = X_norm[1]*100
    #X_norm[2] = X_norm[2]*100
    X_norm = (255.0*X_norm).astype('uint8')
    return X_norm

def Img_normalize3(img):
    img = img.astype('float16')
    for c in range(0, 3):
        Max = np.max(img[:,:,c])
        Min = np.min(img[:,:,c])
        img[:,:,c] = (img[:,:,c]-Min)/(Max-Min)
    img = (255.0*img).astype('uint8')
    return img

def Img_normalize4(img):
    img = img.astype('float16')
    for c in range(0, 3):
        img[:,:,c] = (img[:,:,c]-np.mean(img[:,:,c]))/np.std(img[:,:,c])
    return img

def Img_normalize5(img):
    img = img.astype('float16')
    Max = np.max(img)
    Min = np.min(img)
    for c in range(0, 3):
        #Max = np.max(img[:,:,c])
        #Min = np.min(img[:,:,c])
        img[:,:,c] = (img[:,:,c]-np.mean([Max,Min]))/np.mean([Max,-Min])
        #img[:,:,c] = SubImg_normalize(img[:,:,c])
    return img

class DFLSheet:
    def __init__(self,dflpath,dflname,period,scale,nclass,hsv):
        self.nclass = nclass
        self.period = period
        self.scale = scale
        self.hsv = hsv
        self.dflname = dflname
        
        
        if self.period == 'fresh':
            self.egg_type = ['background','fresh']
            self.nclass = 2
        elif self.period == 'allblue':
            self.egg_type = ['background','allblue','unfer']
            self.nclass = 3
        elif self.period == 'shell':
            self.egg_type = ['background','shell','unfer','dead']
            self.nclass = 4
            
        Img = io.imread(rootpath + dflpath +'\\'+dflname+'\\'+ dflname + '.jpg')
        #io.imsave('Result_img\\hsv_' +dflname +'.jpg',color.rgb2hsv(Img.copy()))
            
        if self.scale==1:
            self.RGBImg = Img
        else:
            RGBImg =  resize(Img, (int(Img.shape[0]*self.scale),int(Img.shape[1]*self.scale)))
            self.RGBImg = Img_normalize_hsv(RGBImg.copy())
            #io.imsave('Result_img\\img_' +dflname +'.jpg',self.RGBImg)
        
        ####################################
        #self.lim_high = 1500
        #self.lim_width = 1500
        self.lim_high = self.RGBImg.shape[0]
        self.lim_width = self.RGBImg.shape[1] 
        #####################################
        
        self.OrigImg = self.RGBImg.copy()
        self.OrigImgLim = self.RGBImg[:self.lim_high,:self.lim_width,:].copy()
        self.RGBImg = self.OrigImgLim.copy()
        
        if self.hsv == 1:
            self.RGBImg = color.rgb2hsv(self.RGBImg.copy())
            #io.imsave('Result_img\\hsv3_' +dflname +'.jpg',Img_normalize3(self.RGBImg.copy()))
            self.RGBImg = Img_normalize_hsv(self.RGBImg.copy())
        else:
            self.RGBImg = Img_normalize(self.RGBImg.copy())
        
        if 1:
            self.RGBImg_max = ndi.maximum_filter(self.RGBImg, size= 500, mode='constant')
            self.RGBImg_min = ndi.minimum_filter(self.RGBImg, size= 500, mode='constant')
            #self.RGBImg = Img_normalize2(self.RGBImg.copy(),self.RGBImg_min,self.RGBImg_max)
            #io.imsave('Result_img\\max_' +dflname +'.jpg',SubImg_normalize(self.RGBImg_max.copy()))
            #io.imsave('Result_img\\norm_' +dflname +'.jpg',SubImg_normalize(self.RGBImg.copy()))
            
       
        self.pointPath = rootpath + dflpath + dflname +'\\'
        self.pointList = []
        self.pointListWhole = []
        self.allImg = []
        self.allLabel = []
        self.eggImg = []
        self.eggLabel = []
        self.eggImgLim = []
        self.eggLabelLim = []
        self.egg_num = np.zeros(self.nclass)
        self.egg_num_lim = np.zeros(self.nclass)
        self.predict_egg = np.zeros(self.nclass)
        
    def read_eggpoint(self,pointPath):
        pointList = []
        pointLimList = []
        lim_high = self.lim_high
        lim_width = self.lim_width
        with open(pointPath, "r") as f:
            for line in f:
                inner_list_str = [elt.strip() for elt in line.split('\t')]
                inner_list_int = list(map(int, inner_list_str))
                inner_list_int[0] = int(inner_list_int[0]*self.scale)
                inner_list_int[1] = int(inner_list_int[1]*self.scale)
                
                if self.period == 'fresh':
                    if inner_list_int[2] <= 6:
                        inner_list_int[2] = 1
                    elif inner_list_int[2] >= 7:
                        inner_list_int[2] = 0
                elif self.period == 'allblue':
                    if inner_list_int[2] == 4 or inner_list_int[2] == 3 or inner_list_int[2] == 5:
                        inner_list_int[2] = 1
                    elif inner_list_int[2] == 1 or inner_list_int[2] == 2 or inner_list_int[2] == 6:
                        inner_list_int[2] = 2
                    elif inner_list_int[2] >= 7:
                        inner_list_int[2] = 0
                else:
                    if inner_list_int[2] == 4 :
                        inner_list_int[2] = 1
                    elif inner_list_int[2] == 6:
                        inner_list_int[2] = 2
                    elif inner_list_int[2] == 5:
                        inner_list_int[2] = 3
                    elif inner_list_int[2] == 7 or inner_list_int[2] == 8:
                        inner_list_int[2] = 0
               # if inner_list_int[2] > 0:  
                pointList.append(inner_list_int)
                if inner_list_int[0] < lim_high and inner_list_int[1] < lim_width:
                    pointLimList.append(inner_list_int)
     
        return pointList
        
    def crop_egg(self,win_size):
        self.pointList = self.read_eggpoint(self.pointPath + 'pointT.txt') 
       
        high = self.RGBImg.shape[0]
        width = self.RGBImg.shape[1]
        half_win = math.floor(win_size/2)
        eggList = []
        eggLabel = []
        for point in self.pointList: 
            x_max = point[1]+half_win 
            x_min = point[1]-half_win
            y_max = point[0]+half_win
            y_min = point[0]-half_win
            x = point[1]
            y = point[0]
            if x_min > 0 and y_min >0 and x_max < high and y_max < width:
                egg = self.RGBImg[x_min:x_max+1,y_min:y_max+1,:].copy()
                #subimg = self.RGBImg[np.max([0,x_min-500]):np.min([x_max+500,high]),np.max([0,y_min-500]):np.min([y_max+500,width]),:].copy()
                
                egg = SubImg_normalize2(egg.copy(), self.RGBImg_min[x,y], self.RGBImg_max[x,y])
                egg[0,0,0] = 1
                #if (point[2]>0 and (x_min+x_max)%1000==0) or point[2]>1:
                    #io.imsave('subimageegg\\'+self.dflname+'_'+str(point[2])+'_'+str(point[1])+'_'+str(point[0])+'.jpg',egg.copy())
                eggList.append(egg.copy())
                eggLabel.append(point[2])
                if point[2] >= 1 and point[2] <= 3:
                    self.egg_num[point[2]] = self.egg_num[point[2]]+1
                else:
                    self.egg_num[0] = self.egg_num[0]+1
                
        self.eggImg = eggList
        self.eggLabel = eggLabel
        self.eggImgLim = eggList
        self.eggLabelLim = eggLabel
        self.egg_num_lim = self.egg_num.copy()
        
    def crop_egg_all(self,win_size):
        self.pointListWhole = self.read_eggpoint(self.pointPath + 'point.txt') 
        high = self.RGBImg.shape[0]
        width = self.RGBImg.shape[1]
        
        half_win = 8 #math.floor(win_size/2)
        eggList = []
        eggLabel = []
        for point in self.pointListWhole: 
            x_max = point[1]+half_win 
            x_min = point[1]-half_win
            y_max = point[0]+half_win
            y_min = point[0]-half_win
            x = point[1]
            y = point[0]
            if x_min > 0 and y_min >0 and x_max < high and y_max < width:
                egg = self.RGBImg[x_min:x_max+1,y_min:y_max+1,:].copy()
                egg = SubImg_normalize2(egg.copy(), self.RGBImg_min[x,y], self.RGBImg_max[x,y])
                egg[0,0,0] = 1
                #if (point[2]>0 and (x_min+x_max)%1000==0) or point[2]>1:
                    #io.imsave('subimage\\'+str(point[2])+'_'+str(point[1])+'_'+str(point[0])+'.jpg',egg.copy())
                
                eggList.append(egg.copy())
                eggLabel.append(point[2])
                if point[2] >= 1 and point[2] <= 3:
                    self.egg_num[point[2]] = self.egg_num[point[2]]+1
                else:
                    self.egg_num[0] = self.egg_num[0]+1
                
        self.eggImg = eggList
        self.eggLabel = eggLabel
        
    def crop_background(self,win_size):
        self.pointList = self.read_eggpoint(self.pointPath + 'pointT.txt') 
        self.pointListWhole = self.read_eggpoint(self.pointPath + 'point.txt') 
        high = self.RGBImg.shape[0]
        width = self.RGBImg.shape[1]
        half_win = 8 #math.floor(win_size/2)
        bgList = []
        bgLabelList = []
        mask_bg = np.zeros(self.RGBImg.shape)
        for point in self.pointListWhole: 
            x_max = point[1]+half_win 
            x_min = point[1]-half_win
            y_max = point[0]+half_win
            y_min = point[0]-half_win
            if x_min > 0 and y_min >0 and x_max < high and y_max < width:
                mask_bg[x_min:x_max+1,y_min:y_max+1] = 1
                      
        step_size = copy.copy(win_size)
        lim_high = self.RGBImg.shape[0]
        lim_width = self.RGBImg.shape[1]
        half_win = math.floor(win_size/2)
        for x in range(half_win,lim_high-half_win,3):
            for y in range(half_win,lim_width-half_win,3*step_size):
                x_max = x+half_win 
                x_min = x-half_win
                y_max = y+half_win
                y_min = y-half_win
                temp_bg = mask_bg[x_min:x_max+1,y_min:y_max+1].copy()
                #if temp_bg.sum() == 0:
                if mask_bg[x,y].sum() == 0:
                    bg = self.RGBImg[x_min:x_max+1,y_min:y_max+1,:].copy()
                    bg = SubImg_normalize2(bg.copy(), self.RGBImg_min[x,y], self.RGBImg_max[x,y])
                    bg[0,0,0] = 1
                    #if (x+y)%1000==0:
                        #io.imsave('subimage\\bg'+str(x)+'_'+str(y)+'.jpg',bg.copy())
                    bgList.append(bg.copy())
                    bgLabelList.append(0)
                    
        num_bg = min(2000,len(bgList))
        idx = range(0,len(bgList))%np.floor(len(bgList)/num_bg).astype('int')==0
        bgImg = [x for (x, y) in zip( bgList, idx) if y]
        bgLabel = [x for (x, y) in zip(  bgLabelList, idx) if y]
        
        self.egg_num[0] = self.egg_num[0]+len(bgLabel)
        self.eggImg = self.eggImg+bgImg
        self.eggLabel = self.eggLabel+bgLabel
        
                
    def crop_all(self,win_size,step_size):
        self.pointListWhole = self.read_eggpoint(self.pointPath + 'point.txt') 
        #lim_high = self.RGBImg.shape[0]
        #lim_width = self.RGBImg.shape[1]
        lim_high = self.lim_high
        lim_width = self.lim_width
        half_win = math.floor(win_size/2)
        eggList = []
        eggPoint = []
        for x in range(half_win,lim_high-half_win,step_size):
            for y in range(half_win,lim_width-half_win,step_size):
                x_max = x+half_win 
                x_min = x-half_win
                y_max = y+half_win
                y_min = y-half_win
                egg = self.RGBImg[x_min:x_max+1,y_min:y_max+1,:].copy()
                egg = SubImg_normalize2(egg.copy(), self.RGBImg_min[x,y], self.RGBImg_max[x,y])
                egg[0,0,0] = 1
                #if (x_min+x_max)%50==0:
                    #io.imsave('suballimage\\x'+'_'+str(x)+'_'+str(y)+'.jpg',egg.copy())
                
                eggList.append(egg.copy())
                eggPoint.append([x,y])
                
        self.allImg = eggList 
        self.allPoint = eggPoint 
                 
        X = np.array(self.pointListWhole)[:,0]
        Y = np.array(self.pointListWhole)[:,1]
        label = np.array(self.pointListWhole)[:,2]
        egg_num = np.zeros(self.nclass)
        for c in range(0,self.nclass):
            egg_num[c] = (label==c).sum()
        self.egg_num = egg_num   
        
        egg_num_lim = np.zeros(self.nclass)
        for c in range(0,self.nclass):
            egg_num_lim[c] = ((label==c) & (X <= lim_high) & (Y <= lim_width)).sum()
        self.egg_num_lim = egg_num_lim   
        
    
class DFLSheetDataSet:
    
    def __init__(self,params):
        self.period = params['period']
        self.scale = params['scale']
        self.window_size = params['window_size']
        self.egg_num = pd.DataFrame() 
        self.hsv = params['hsv']
        if self.period == 'fresh':
            print("Fresh egg sheets are loaded")
            self.nclass = 2
            self.rootpath = 'F\\'
            self.dflsheet_train = ['F4','F12','F16']
            self.dflsheet_blind = ['F1','F5','F6','F13','F14','F15']
            self.egg_type = ['background','fresh']
            self.ratioEgg = [0.55 ,0.50]
            
        elif self.period == 'allblue':
            print("All-blue egg sheets are loaded")
            self.nclass = 3
            self.rootpath = 'A\\'
            self.dflsheet_train = ['A4','F7s','F10s']#,'F2p','A1''A2',
            self.egg_type = ['background','allblue','unfer']
            self.dflsheet_blind = ['F3p','F8s','F9s','F11s']
            self.ratioEgg = [3. ,1., .8] #[1. ,1., 1.2]
            
        elif self.period == 'shell':
            print("Shell egg sheets are loaded")
            self.nclass = 4
            self.rootpath = 'S\\'
            #self.dflsheet_train = ['S23a','S36s','S4']#'A1','S2',
            self.dflsheet_train = ['S23a','S36s','S4','S33s']
            self.egg_type = ['background','shell','unfer','dead']
            self.dflsheet_blind = ['S5','S6','S8','S9','S10','S11p','S12',
                                   'S13p','S14a','S15a','S16a','S17a','S18a',
                                   'S19a','S20a','S21a','S22a','S24a','S25a',
                                   'S26a','S27a','S28a','S29a','S30a','S31p','S35s']
            self.ratioEgg = [3. ,3., 0.45 ,0.5] #[3. ,3., 0.25 ,0.3]
        else:
            print('\"'+ params['period'] +'\" is not defind. please check the DFL period!!!')
        
        import pickle
        pickle.dump(params,open(params['dataname'] +'/param.pkl',"wb"))
        params = pickle.load(open(params['dataname'] +'/param.pkl', 'rb'))
   
    
    def generate_signatureLibraly(self):
        self.create_signatureLibraly()
        self.balance_dataset()
    
    def create_signatureLibraly(self):
        self.dfl = []
        print("======= Training DFL Images ========")
        for i in range(0,len(self.dflsheet_train)):
            dfl = []
            dfl = DFLSheet(self.rootpath,self.dflsheet_train[i],self.period,self.scale,self.nclass,self.hsv)
            print(self.dflsheet_train[i]+' loading was finished')
            dfl.crop_egg(self.window_size)
            dfl.crop_background(self.window_size)
            line_num = pd.DataFrame([dfl.egg_num], columns = dfl.egg_type, index = [self.dflsheet_train[i]])
            self.egg_num = self.egg_num.append(line_num)
            self.dfl.append(dfl)
            
    def create_all_subimg_dfl(self,step_size,dflsheet):
        self.dfl = []
        self.egg_num = pd.DataFrame()
        print("======= Create Sub-DFL Images ========")
        for i in range(0,len(dflsheet)):
            dfl = []
            dfl = DFLSheet(self.rootpath,dflsheet[i],self.period,self.scale,self.nclass,self.hsv)
            print(dflsheet[i]+' sub-image is done')
            dfl.crop_all(self.window_size,step_size)
            line_num = pd.DataFrame([dfl.egg_num], columns = dfl.egg_type, index = [dflsheet[i]])
            self.egg_num = self.egg_num.append(line_num)
            self.dfl.append(dfl)
            
    def create_egg_num_file(self):
        print("======= Count Egg on DFL Images ========")
        for i in range(0,len(self.dflsheet_train)):
            dfl = []
            dfl = DFLSheet(self.rootpath,self.dflsheet_train[i],self.period,self.scale,self.nclass,self.hsv)
            print(self.dflsheet_train[i]+' loading was counted')
            dfl.crop_egg_all(self.window_size)
            line_num = pd.DataFrame([dfl.egg_num], columns = dfl.egg_type, index = [self.dflsheet_train[i]])
            self.egg_num = self.egg_num.append(line_num)
            
        for i in range(0,len(self.dflsheet_blind)):
            dfl = []
            dfl = DFLSheet(self.rootpath,self.dflsheet_blind[i],self.period,self.scale,self.nclass,self.hsv)
            print(self.dflsheet_blind[i]+' loading was counted')
            dfl.crop_egg_all(self.window_size)
            line_num = pd.DataFrame([dfl.egg_num], columns = dfl.egg_type, index = [self.dflsheet_blind[i]])
            self.egg_num = self.egg_num.append(line_num)
        
    
    def balance_dataset(self):
        min_egg = int(self.egg_num.min(axis=0).min(axis=0))
        print(self.egg_num)
        train_sigImg = []
        train_sigLabel = []
        blind_sigImg = []
        blind_sigLabel = []
        #self.egg_num.iloc[row_idx].mi_L0
        for i in range(0,len(self.dflsheet_train)):
            sigImg = self.dfl[i].eggImg
            sigLabel = self.dfl[i].eggLabel
            for c in range(0,self.nclass):
                num_egg_this = (np.array(sigLabel) == c).sum()
                min_egg_num = int(min(num_egg_this, max(250*self.ratioEgg[c],min_egg*self.ratioEgg[c]))) 
                idx = (np.array(sigLabel) == c)
                selected_imgs = [x for (x, y) in zip(sigImg, idx) if y]
                selected_labels = [x for (x, y) in zip(sigLabel, idx) if y]
                
                train_sigImg += copy.copy(selected_imgs[0:min_egg_num-1])
                train_sigLabel += copy.copy(selected_labels[0:min_egg_num-1])
                blind_sigImg += copy.copy(selected_imgs[min_egg_num:min(6000,num_egg_this)])
                blind_sigLabel += copy.copy(selected_labels[min_egg_num:min(6000,num_egg_this)])
                
        self.train_sigImg = copy.copy(np.array(train_sigImg))
        self.train_sigLabel = copy.copy(np.array(train_sigLabel))
        self.blind_sigImg = copy.copy(np.array(blind_sigImg))
        self.blind_sigLabel = copy.copy(np.array(blind_sigLabel))
        
        self.sig_egg_num = pd.DataFrame(columns = self.dfl[i].egg_type)
        egg1 = np.zeros(self.nclass)
        egg2 = np.zeros(self.nclass)
        for c in range(0,self.nclass):
            egg1[c] = (self.train_sigLabel==c).sum()
            egg2[c] =(self.blind_sigLabel==c).sum()
        
        eggTrain = pd.DataFrame([egg1],columns = self.dfl[i].egg_type, index = ['Traing set'] )
        eggBlind = pd.DataFrame([egg2],columns = self.dfl[i].egg_type, index = ['Blind set'] )
        self.sig_egg_num = self.sig_egg_num.append(eggTrain)
        self.sig_egg_num = self.sig_egg_num.append(eggBlind)
        

        
        
        
        
        
        
        
        
        