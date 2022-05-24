from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from catboost import CatBoostRegressor
from .utils import *

class cde_reg:
    '''
    Model for Y|X=x. We assume that Y|X=x ~ Normal(f(x),sigma2), where f(x) is a function of the features.
    (This is class in Scikit-Learn style)
    '''
    
    def __init__(self, boost=True, validation_split=.1, cat_features=None, cv=5):
        
        '''
        Input:  (i)   boost: if TRUE, we use CatBoost as regressor - otherwise, we use linear regression (OLS or Ridge);
                (ii)  validation_split: portion of the training data used to early stop CatBoost and to estimate sigma2 - this parameter is not used if 'boost'==FALSE;
                (iii) cat_features: list containing all categorical features indices - used only if 'boost'==TRUE;
                (iv)  cv: number of CV folds used to validade Ridge regression classifier - this parameter is not used if 'boost'==TRUE. If cv==None, then we use the default Scikit-Learn config. for LinearRegression;
        '''
        
        self.boost=boost
        self.validation_split=validation_split
        self.cat_features=cat_features
        self.cv=cv
    
    def fit(self, X, y, random_seed=None):
        
        '''
        Function that fits the conditional density model;

        Input:  (i)   X: Pandas Dataframe of features - use the 'prep_data' function to prepare your data;
                (ii)  y: Pandas Dataframe of label - use the 'prep_data' function to prepare your data;

        Output: None
        '''
        
        n=X.shape[0]

        if self.boost:   
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_split, random_state=random_seed)
        
            self.model =  CatBoostRegressor(cat_features=self.cat_features,
                                            thread_count=-1,
                                            random_seed=random_seed)

            self.model.fit(X_train, y_train,
                      verbose=False,
                      eval_set=(X_val, y_val),
                      early_stopping_rounds = 100)

            e = self.model.predict(X_val).squeeze()-np.array(y_val).squeeze()
            self.s2=np.var(e)


        else:
            if self.cv==None:
                self.model = LinearRegression().fit(X, y)
                e=(self.model.predict(X).squeeze()-np.array(y).squeeze())
                n=e.shape[0]
                self.s2=(np.sum(e**2)/(n-X.shape[1]-1))
                
            else: 
                alphas=np.logspace(-4,4,10)
                self.model = RidgeCV(alphas=alphas, cv=self.cv).fit(X, y)
                e=(self.model.predict(X).squeeze()-np.array(y).squeeze())
                n=e.shape[0]
                self.s2=(np.sum(e**2)/(n-X.shape[1]-1))    
    
    def sample(self, X): 
        
        '''
        Function that samples Y|X=x using the probabilistic model Y|X=x ~ Normal(f(x),sigma2)) with fitted model
        '''
        
        return pd.DataFrame(np.random.normal(self.model.predict(X), self.s2**.5))
    
    
class cde_class:
    '''
    Model for P(Y=1|X), that is, binary probabilistic classifier. See that Y|X=x ~ Multinomial(n=1,p(x)), where p(.) is a function of the features.
    (This is class in Scikit-Learn style)
    '''
    
    def __init__(self, boost=True, validation_split=.1, cat_features=None, cv=5):
        
        '''
        Input:  (i)   boost: if TRUE, we use CatBoost as classifier - otherwise, we use a not regularized logistic regression;
                (ii)  validation_split: portion of the training data (Zs,Zt) used to early stop CatBoost - this parameter is not used if 'boost'==FALSE;
                (iii) cat_features: list containing all categorical features indices - used only if 'boost'==TRUE;
                (iv)  cv: number of CV folds used to validade the logistic regression classifier - this parameter is not used if 'boost'==TRUE;
        '''
        
        self.boost=boost
        self.validation_split=validation_split
        self.cat_features=cat_features
        self.cv=cv
    
    def fit(self, X, y, random_seed=None):
        
        '''
        Function that fits the classification model in order to estimate P(Y=1|X);

        Input:  (i)   X: Pandas Dataframe of features - use the 'prep_data' function to prepare your data;
                (ii)  y: Pandas Dataframe of label - use the 'prep_data' function to prepare your data;

        Output: None
        '''
        
        n=X.shape[0]
        y=get_classes(y) #Transforming dummies to a unique array y (needed to train the models)
        
        if self.boost:   
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_split, stratify=y, random_state=random_seed)

            self.model =  CatBoostClassifier(loss_function = 'MultiClass',
                                             cat_features=self.cat_features,
                                             thread_count=-1,
                                             random_seed=random_seed)

            self.model.fit(X_train, y_train,
                           verbose=False,
                           eval_set=(X_val, y_val),
                           early_stopping_rounds = 100)
         
        else:
            if self.cv==None:
                self.model = LogisticRegression(solver='liblinear', random_state=random_seed).fit(X, y)
            else: 
                self.model = LogisticRegressionCV(cv=self.cv, scoring='neg_log_loss', solver='liblinear', 
                                                  random_state=random_seed).fit(X, y)
                
            
    def sample(self, X): 
        
        '''
        Function that samples Y|X=x using the probabilistic fitted model \hat{P}(Y=y|X=x);
        '''
        ps=self.model.predict_proba(X)
        samp=np.array([np.argmax(np.random.multinomial(1, p)) for p in ps])
        samp=pd.DataFrame(samp)
        
        return get_dummies(samp)
    