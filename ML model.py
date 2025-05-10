import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import xgboost as xgb
import shap
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split as TTS
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from pyod.utils.data import get_outliers_inliers
from sklearn.utils import compute_sample_weight
import re

#from chord import Chord
plt.rc('font',family='Arial') 


os.chdir('D:/Work/GPI/Data/USA/')

def xgbmodel(df, max_depth, min_child_weight, colsample_bytree, subsample, change=False):  

    X = df.drop(df.iloc[:,0:9],axis=1)
    X = X.drop(X[['stcofips']],axis=1) #,'chickens','horses','mcows','othercows'
#    X = X.drop(X[['P_MAN_TOT','farmp','nonfp','stcofips']],axis=1) #,'chickens','horses','mcows','othercows'
                  #'sheep','barley','corns','cotton','dbeans',
                  #'hayalf','oats','rice','sorghum-grain','sorghums','wheat'
    #X = X.drop(X[['uniqid','N_fer_use','year']],axis=1)
    #X = X.drop(X[['uniqid','N_fer_use']],axis=1)

#    X_, vif_= vif(X,15)   
    y = df['ResultMeas'].copy()
    if change == True:
        per = 1/2
        cut = 0
        for i in range(len(y)):
            if np.sort(y)[i] < 0 and np.sort(y)[i+1] >= 0:
                cut = i
        pos = 1 # np.mean(np.sort(y)[cut+1:len(y)-1])
        neg = -1 # np.mean(np.sort(y)[0:cut])
        for i in range(y.shape[0]):
            if y.iat[i] <= neg*per:
                y.iat[i] = 0
            elif abs(y.iat[i]) <= pos*0.1:
                y.iat[i] = 1
            elif y.iat[i] <= pos*per:
                y.iat[i] = 2
            elif y.iat[i] <= pos*per*2:
                y.iat[i] = 3
            else: y.iat[i] = 4
        #plt.hist(y)
    else:
        #perc = np.percentile(y.values,(20,40,60,80),interpolation='midpoint')
        for i in range(y.shape[0]):
            if y.iat[i] <= 2:
                y.iat[i] = 0
            elif y.iat[i] <= 4:
                y.iat[i] = 1
            elif y.iat[i] <= 10:
                y.iat[i] = 2
#            elif y.iat[i] <= 20:
#                y.iat[i] = 3
            else: y.iat[i] = 3

    Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=6)
    
    alg = xgb.XGBClassifier(
    #alg = xgb.XGBRegressor(
        learning_rate =0.01,
        n_estimators=50000,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=0,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective= 'multi:softprob',
        num_class=4,
        nthread=4,
        seed=0)
        #seed=6)
    
    cv_folds=5
    early_stopping_rounds=1000
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(Xtrain.values, label=Ytrain.values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
        metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
    alg.set_params(n_estimators=cvresult.shape[0])
    print(cvresult.shape[0])
    #Fit the algorithm on the data
    alg.fit(Xtrain, Ytrain, eval_metric='mlogloss',sample_weight = compute_sample_weight('balanced', Ytrain))
    

    return alg, Xtrain, Xtest, df['ResultMeas'], Ytrain, Ytest, cvresult, df#, vif_


#year = 1985
year = 2010
df = pd.read_csv('CM/CoA/Samples/lag/'+str(year)+'_v2.csv')
(test, Xtrain, Xtest, y,
 Ytrain, Ytest, cvresult, df) = xgbmodel2(df, max_depth, min_child_weight, colsample_bytree, subsample, change=False)
