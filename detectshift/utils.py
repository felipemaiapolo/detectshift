from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import OneHotEncoder

def check_array_X(X, name="X", dim=2): 
    if not type(X) in [np.ndarray,pd.DataFrame] or len(X.shape)!=dim:
            raise ValueError(name+" must be a Pandas data frame or a {:}-dimensional Numpy array.".format(dim))       
def check_array_y(X, name="y", dim=2): 
    if not type(X) in [np.ndarray,pd.DataFrame] or len(np.array(X).squeeze().shape)!=dim:
            raise ValueError(name+" must be a Pandas data frame or a {:}-dimensional Numpy array.".format(dim))   
   
def get_dummies(y):
    '''
    One hot encoder
    '''
    if type(y)==np.ndarray:
        y=y.reshape(-1,1)
        
    onehot = OneHotEncoder(drop='first', sparse=False).fit(y)
    out=pd.DataFrame(onehot.transform(y))
    out.columns = ['y'+str(j) for j in range(out.shape[1])]
    return out.astype('int32')

def get_classes(y):
    '''
    One hot decoder
    '''
    ind=np.sum(y,axis=1)==0
    classes=np.argmax(np.array(y), axis=1)+1
    classes[ind]=0
    return classes.astype('int32')

def prep_data(Xs, ys, Xt, yt, test=.1, task=None, random_state=42):
    
    '''
    Function that gets data and prepare it to run the tests
    
    Input:  (i)   Xs and Xt: 2d-numpy array or Pandas Dataframe containing features from the source and target domain;
            (ii)  ys and yt: 1d-numpy array or 1-column Pandas Dataframe containing labels. If task=='class', then ys and yt must contain all labels [0,1,...,K-1], where K is the number of classes;
            (iii) test: fraction of the data going to the test set;
            (iv)  task: 'reg' or 'class' for regression or classification;
            (v)   random_state: seed used in the data splitting
            
    Output: Xs_train, Xs_test, ys_train, ys_test, Zs_train, Zs_test
            Xt_train, Xt_test, yt_train, yt_test, Zt_train, Zt_test
    '''
    
    ##Checking task
    if task not in ['reg','class']:
        raise ValueError("'task' must be in ['reg','class'].")
        
    #Checking X and y
    check_array_X(Xs, name="Xs", dim=2)
    check_array_y(ys, name="ys", dim=1)
    check_array_X(Xt, name="Xt", dim=2)
    check_array_y(yt, name="yt", dim=1)
    
    ##Checking if ys and yt have the same values
    if task == 'class':
        if not (np.unique(np.array(ys.astype('int32')))==np.unique(np.array(yt.astype('int32')))).all():
            raise ValueError("ys and yt must be composed of the same classes.")

    ##Output
    out1 = prep_data_aux(Xs, ys, test=test, task=task, random_state=random_state)
    out2 = prep_data_aux(Xt, yt, test=test, task=task, random_state=random_state)
    return out1+out2
        
def prep_data_aux(X, y, test=.1, task=None, random_state=42):

    ##Checking y values
    if task == 'class':
        y=np.array(y.astype('int32')).squeeze()
        if not np.unique(y).tolist()==list(range(0,np.max(y)+1)):
            raise ValueError("ys and yt must be composed of classes from 0 to K-1, where K is total number of classes.")
    
    ##Converting to Pandas Data Frame
    X, y = pd.DataFrame(np.array(X)), pd.DataFrame(np.array(y)) 
    
    if task == 'class':
        y = y.astype('int32')
        
    ##Applying onehot encoding
    if task=='class':
        y = get_dummies(y)
    
    ##Renaming columns
    X.columns = ['x'+str(j) for j in range(X.shape[1])]
    y.columns = ['y'+str(j) for j in range(y.shape[1])]
    
    ##Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=random_state)
    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True) 
    
    ##Checking if y_train and y_test have the same values
    if task == 'class':
        k_train=np.unique(np.array(get_classes(y_train).astype('int32'))).squeeze().shape[0]
        k_test=np.unique(np.array(get_classes(y_test).astype('int32'))).squeeze().shape[0]
        if not (k_train==k_test):
            raise ValueError("y_train and y_test must be composed by the same classes. Try increasing test proportion.")
      
    ##Creating Z's
    Z_train = pd.concat([X_train, y_train], axis=1) 
    Z_test = pd.concat([X_test, y_test], axis=1) 
    
    return X_train, X_test, y_train, y_test, Z_train, Z_test

def KL_multinomial(ys, yt):
    
    '''
    Function that estimate the Symmetrized DKL for two Bernoulli iid vectors
    
    Input:  (i)  ys: 1d-numpy array of [0,1,...,K] coming from the source population;
            
    Output: (i) estimate of the DKL of the two categorical r.v.s that generate the vectors
    '''
    
    kts=0
    Y=np.unique(np.array(ys))
    
    for y in Y:
        pt=np.mean(np.array(yt).squeeze()==y)
        ps=np.mean(np.array(ys).squeeze()==y)
        kts+=pt*np.log(pt/ps)
        
    return kts 

class KL:
    
    '''
    Model to estimate the DKL using the classification approach to density ratio estimation
    (this is class in Scikit-Learn style)
    '''
    
    def __init__(self, boost=True, validation_split=.1, cat_features=None, cv=5):
        
        '''
        Input:  (i)   boost: if TRUE, we use CatBoost as classifier - otherwise, we use logistic regression;
                (ii)  validation_split: portion of the training data (Zs,Zt) used to early stop CatBoost - this parameter is not used if 'boost'==FALSE;
                (iii) cat_features: list containing all categorical features indices - used only if 'boost'==TRUE;
                (iv)  cv: number of CV folds used to validate the logistic regression classifier - this parameter is not used if 'boost'==TRUE. Hyperparameter values tested are specified in Scikit-Learn's "LogisticRegressionCV" class. If cv==None, then we use the default Scikit-Learn config. for LogisticRegression;
        '''
        
        self.boost=boost
        self.validation_split=validation_split
        self.cat_features=cat_features
        self.cv=cv
  
    def fit(self, Zs, Zt, random_state=0):
        
        '''
        Function that fits the classification model in order to estimate the density ratio w=p_t/p_s (target dist. over source dist.)

        Input:  (i)   Zs: bidimensional array or Pandas DataFrame (usually X or (X,y)) coming from the source distribution - use the 'prep_data' function to prepare your data;
                (ii)  Zt: bidimensional array or Pandas DataFrame (usually X or (X,y)) coming from the target distribution - use the 'prep_data' function to prepare your data;
                (iii) random_state: seed used in the data splitting and model training
        Output: None
        '''
        
        self.nt, self.ns = Zt.shape[0], Zs.shape[0]
        
        Xw = pd.concat([Zt,Zs], axis=0) 
        yw = np.hstack((np.ones(self.nt),np.zeros(self.ns)))
        
        if self.boost: 
            Xw_train, Xw_val, yw_train, yw_val = train_test_split(Xw, yw, test_size=self.validation_split, random_state=random_state)

            self.model =  CatBoostClassifier(loss_function = 'Logloss',
                                             cat_features=self.cat_features,
                                             thread_count=-1,
                                             random_seed=random_state)

            self.model.fit(Xw_train, yw_train,
                           verbose=False,
                           eval_set=(Xw_val, yw_val),
                           early_stopping_rounds = 100)
         
        else:           
            if self.cv==None:
                self.model = LogisticRegression(solver='liblinear', random_state=random_state).fit(Xw, yw)
            else: 
                self.model = LogisticRegressionCV(cv=self.cv, scoring='neg_log_loss', solver='liblinear', 
                                                  random_state=random_state).fit(Xw, yw)
                
            

    def predict_w(self, Z, eps=10**-10):
        
        '''
        Function that predicts the density ratio w=p_t/p_s (target dist. over source dist.)

        Input:  (i) Z: bidimensional array or Pandas DataFrame (usually X or (X,y)) coming from the source distribution;
        
        Output: (ii) An array containing the predicted density ratio w=p_t/p_s for each row of Z
        '''
        
        p = self.model.predict_proba(Z)[:,1]
        prior_ratio = self.ns/self.nt
        return prior_ratio*((p+eps)/(1-p+eps))

    def predict(self, Zt, eps=10**-10):
        
        '''
        Function that infers the DKL of the distirbutions that generated Zs and Zt

        Input:  (i) Zt: bidimensional array or Pandas DataFrame (usually X or (X,y)) coming from the target distribution;
        
        Output: (i) Point estimate of DKL
        '''
        
        predt=self.predict_w(Zt, eps)
        
        return np.mean(np.log(predt)) 
    


    
   