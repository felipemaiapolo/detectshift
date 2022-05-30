# ***DetectShift*** - Dataset shift diagnostics in Python

This package is based on the ideas presented in

[**Polo, F. M., Izbicki, R., Lacerda Jr, E. G., Ibieta-Jimenez, J. P., & Vicente, R. (2022). A unified framework for dataset shift diagnostics. arXiv preprint arXiv:2205.08340.**](https://arxiv.org/abs/2205.08340). 

If you use our package in your academic work, please cite us in the following way

    @article{polo2022unified,
      title={A unified framework for dataset shift diagnostics},
      author={Polo, Felipe Maia and Izbicki, Rafael and Lacerda Jr, Evanildo Gomes and Ibieta-Jimenez, Juan Pablo and Vicente, Renato},
      journal={arXiv preprint arXiv:2205.08340},
      year={2022}
    }

In case you have any question or suggestion, please get in touch sending us an e-mail in *felipemaiapolo@gmail.com*.

--------------

## Summary

0. [Quick start (demo)](#0)
1. [Introduction](#1)
    1. [Installing and loading package ](#1.1)
    2. [Steps to use](#1.2)
2. [Modules](#2)
    1. [`tools`](#2.1)
    2. [`cdist`](#2.2)
    3. [`tests`](#2.3)


--------------

<a name="0"></a>
## 0\. Quick start (demo)

Below are the links to some demonstrations on how to use *DetectShift* in practice:

- **[Binary classification:](https://colab.research.google.com/github/felipemaiapolo/detectshift/blob/main/demo/Classification1.ipynb)** In this notebook, we showcase an use example of dataset shift diagnostics when the response variable $Y$ is binary.

- **[Multinomial classification:](https://colab.research.google.com/github/felipemaiapolo/detectshift/blob/main/demo/Classification2.ipynb)** In this notebook, we showcase an use example of dataset shift diagnostics when the response variable $Y$ is discrete with more than 2 values.

- **[Regression:](https://colab.research.google.com/github/felipemaiapolo/detectshift/blob/main/demo/Regression1.ipynb)** In this notebook, we showcase an use example of dataset shift diagnostics when the response variable $Y$ is continuous.

- **[Regression with categorical features:](https://colab.research.google.com/github/felipemaiapolo/detectshift/blob/main/demo/Regression2.ipynb)** In this notebook, we showcase an use example of dataset shift diagnostics when the response variable $Y$ is quantitative but discrete (with many different values). Also, in this example, we make use of categorical features, exploiting [Catboost](https://catboost.ai/) functionality.




--------------

<a name="1"></a>
## 1\. Introduction 
*DetectShift* aims to quantify and test which types of dataset shift occur in a dataset. 

If $Q$ denotes the source distribution and $P$ denotes the target distribution, the null hypotheses we want to test are:

- **[Total Dataset Shift]:** $H_{0,\text{D}}:P_{X,Y}=Q_{X,Y}$
- **[Covariate Shift]:** $H_{0,\text{C}}:P_{X}=Q_{X}$
- **[Label Shift]:** $H_{0,\text{L}}:P_{Y}=Q_{Y}$
- **[Concept Shift 1]:** $H_{0,\text{C1}}:P_{X|Y}=Q_{X|Y}$
- **[Concept Shift 2]:** $H_{0,\text{C2}}: P_{Y|X}=Q_{Y|X}$

The test statistic used is a KL divergence estimator while the p-values are obtained through simulation (permutation/randomization) tests.

Check [**our paper**](https://arxiv.org/abs/2205.08340) for more details and precise language. 

<a name="1.1"></a>
### 1.1\.  Installing and loading package 

You can install our package running the following command on terminal
``` :sh
$ pip install git+https://github.com/felipemaiapolo/detectshift
```

You can load *DetectShift* in the following way

```python
import detectshift as ds
```

<a name="1.2"></a>
### 1.2\.  Steps to use

If you take a look at *DetectShift* [demonstrations](#0), you will realize that following some steps to use *DetectShift* are needed. The steps are:

1. Loading *DetectShift*;
2. Preparing data with `prep_data` function;
3. Instantiating and training models to estimate KL divergence using `KL` class;
4. Instantiating and training models to estimate the conditional distirbution of $Y|X$ using `cdist` module (in case of testing for concept shift of type 2;
5. Testing different types of shift using the `tests` module. 


--------------

<a name="2"></a>
## 2\. Modules

<a name="2.1"></a>
### 2.1\.  `tools`

Module containing general tools. Main functionalities include a function to prepare data and a class of models to estimate KL divergence.

#### 2.1.1\. Function: `prep_data(Xs, ys, Xt, yt, test=.1, task=None, random_state=42)` 
   
    Function that gets data and prepare it to run the tests
    
    Input:  (i)   Xs and Xt: 2d-numpy array or Pandas Dataframe containing features from the source and target domain;
            (ii)  ys and yt: 1d-numpy array or 1-column Pandas Dataframe containing labels. If task=='class', then ys and yt must contain all labels [0,1,...,K-1], where K is the number of classes;
            (iii) test: fraction of the data going to the test set;
            (iv)  task: 'reg' or 'class' for regression or classification;
            (v)   random_state: seed used in the data splitting
            
    Output: Xs_train, Xs_test, ys_train, ys_test, Zs_train, Zs_test
            Xt_train, Xt_test, yt_train, yt_test, Zt_train, Zt_test
            
Here Z stands for (X,y).
            
            
            
#### 2.1.2\. Class: `KL`

Model to estimate the DKL using the classification approach to density ratio estimation (this is class in Scikit-Learn style).
   
- `__init__(self, boost=True, validation_split=.1, cat_features=None, cv=5)`

        Input:  (i)   boost: if TRUE, we use CatBoost as classifier - otherwise, we use logistic regression;
                (ii)  validation_split: portion of the training data (Zs,Zt) used to early stop CatBoost - this parameter is not used if 'boost'==FALSE;
                (iii) cat_features: list containing all categorical features indices - used only if 'boost'==TRUE;
                (iv)  cv: number of CV folds used to validate the logistic regression classifier - this parameter is not used if 'boost'==TRUE. Hyperparameter values tested are specified in Scikit-Learn's "LogisticRegressionCV" class. If cv==None, then we use the default Scikit-Learn config. for LogisticRegression;


- `fit(self, Zs, Zt, random_state=0)`

       Function that fits the classification model in order to estimate the density ratio w=p_t/p_s (target dist. over source dist.)

       Input:  (i)   Zs: bidimensional array or Pandas DataFrame (usually X or (X,y)) coming from the source distribution - use the 'prep_data' function to prepare your data;
               (ii)  Zt: bidimensional array or Pandas DataFrame (usually X or (X,y)) coming from the target distribution - use the 'prep_data' function to prepare your data;
               (iii) random_state: seed used in the data splitting and model training

       Output: None
   
   
- `predict_w(self, Z, eps=10**-10)`

       Function that predicts the density ratio w=p_t/p_s (target dist. over source dist.)

       Input:  (i) Z: bidimensional array or Pandas DataFrame (usually X or (X,y)) coming from the source distribution;

       Output: (ii) An array containing the predicted density ratio w=p_t/p_s for each row of Z


- `predict(self, Zt, eps=10**-10)` 
       
       Function that infers the DKL of the distirbutions that generated Zs and Zt

       Input:  (i) Zt: bidimensional array or Pandas DataFrame (usually X or (X,y)) coming from the target distribution;

       Output: (i) Point estimate of DKL


<a name="2.2"></a>
### 2.2\.  `cdist`

Module containing classes of models to estimate the conditional distribution of $Y|X$.

#### 2.2.1\. Class: `cde_reg`

Model for Y|X=x. We assume that Y|X=x ~ Normal(f(x),sigma2), where f(x) is a function of the features. (This is class in Scikit-Learn style).

*The user could adapt this class in order to use different models than the Normal one.*

- `__init__(self, boost=True, validation_split=.1, cat_features=None, cv=5)`
        
        Input:  (i)   boost: if TRUE, we use CatBoost as regressor - otherwise, we use linear regression (OLS or Ridge);
                (ii)  validation_split: portion of the training data used to early stop CatBoost and to estimate sigma2 - this parameter is not used if 'boost'==FALSE;
                (iii) cat_features: list containing all categorical features indices - used only if 'boost'==TRUE;
                (iv)  cv: number of CV folds used to validade Ridge regression classifier - this parameter is not used if 'boost'==TRUE. If cv==None, then we use the default Scikit-Learn config. for LinearRegression;


- `fit(self, X, y, random_seed=None)`
        
        Function that fits the conditional density model;

        Input:  (i)   X: Pandas Dataframe of features - use the 'prep_data' function to prepare your data;
                (ii)  y: Pandas Dataframe of label - use the 'prep_data' function to prepare your data;
        Output: None
 
- `sample(self, X)`
        
        Function that samples Y|X=x using the probabilistic model Y|X=x ~ Normal(f(x),sigma2)) with fitted model
        
        Input:  (i)   X: Pandas Dataframe of features - use the 'prep_data' function to prepare your data;
        Output: (i)   Samples from the conditional distribution.
      
      
#### 2.2.2\. Class: `cde_class`

Model for P(Y=y|X), that is, binary probabilistic classifier. See that Y|X=x ~ Multinomial(n=1,p(x)), where p(.) is a function of the features. (This is class in Scikit-Learn style)

    
- `__init__(self, boost=True, validation_split=.1, cat_features=None, cv=5)`
        

        Input:  (i)   boost: if TRUE, we use CatBoost as classifier - otherwise, we use a not regularized logistic regression;
                (ii)  validation_split: portion of the training data (Zs,Zt) used to early stop CatBoost - this parameter is not used if 'boost'==FALSE;
                (iii) cat_features: list containing all categorical features indices - used only if 'boost'==TRUE;
                (iv)  cv: number of CV folds used to validade the logistic regression classifier - this parameter is not used if 'boost'==TRUE;

    
- `fit(self, X, y, random_seed=None)`
        
        Function that fits the classification model in order to estimate P(Y=y|X);
        
        Input:  (i)   X: Pandas Dataframe of features - use the 'prep_data' function to prepare your data;
                (ii)  y: Pandas Dataframe of label - use the 'prep_data' function to prepare your data;
                
        Output: None
   
            
- `sample(self, X)` 
        
        Function that samples Y|X=x using the probabilistic fitted model \hat{P}(Y=y|X=x);
        
        Input:  (i)   X: Pandas Dataframe of features - use the 'prep_data' function to prepare your data;
        
        Output: (i)   Samples from the conditional distribution.
        
        
<a name="2.3"></a>
### 2.3\.  `tests`

Module containing hypotheses tests functions.

#### 2.3.1\. Function: `ShiftDiagnostics(Xs_test, ys_test, Xt_test, yt_test, totshift_model, covshift_model, labshift_model, cd_model, task, n_bins=10, B=500, verbose=True)`

    Function that returns results for all the tests
    
    Input:  (i)    Xs_test and ys_test: Two Pandas dataframes with X and y from the source population - use the 'prep_data' function to prepare your data;
            (ii)   Xt_test and yt_test: Two Pandas dataframes with X and y from the target population - use the 'prep_data' function to prepare your data;
            (iii)  totshift_model: KL model used to estimate the Dkl between the two joint distributions of (X,y) (trained using training set);
            (iv)   covshift_model: KL model used to estimate the Dkl between the two marginal distributions of features X (trained using training set);
            (v)    labshift_model: KL model used to estimate the Dkl between the two marginal distributions of labels y (trained using training set) - you can set labshift_model=None if task=='class' and, in this case, the function will call "KL_multinomial" as estimator;
            (vi)   cd_model: conditional density model equiped with 'sample' function. See documentation for more details;
            (vii)  task: 'class' or 'reg' for classification or regression;
            (viii) n_bins: number of bins if performing regression task. If task=='reg', this function will evenly bin ys, yt based on y=(ys,yt) quantiles. We use binning only to get the p-value and we report the original KL estimate;
            (ix)   B: number of permutations used to calculate p-value;
            
    Output: (i) Dictionary containing the pvalues, the estimates of the shifts (Dkl's) and the permutations values;


#### 2.3.2\. Function: `Permut(Zs, Zt, shift_model, B=500, verbose=True)` 
    
    Function that returns the permutation p-values for testing H0 (Pt=Ps) for distributions of Z, where Z can be X, y, or (X,y)
    
    Input:  (i)   Zs: Pandas dataframe with Z (typically X or (X,y)) from the source population (test set prefered) - use the 'prep_data' function to prepare your data;
            (ii)  Zt: Pandas dataframe with Z (typically X or (X,y)) from the target population (test set prefered) - use the 'prep_data' function to prepare your data;
            (iii) shift_model: KL model used to estimate the Dkl between Pt and Ps (trained using training set);
            (iv)  B: number of permutations used to calculate p-value;
            
    Output: (i) Dictionary containing the pvalue, the estimate of the shift (Dkl's) and the permutations values;
    
    
#### 2.3.3\. Function: `PermutDiscrete(Zs, Zt, B=500, verbose=True)` 
    
    Function that returns the permutation p-values for testing H0 (Pt=Ps) for distributions of Z (typically y in classification problems). **We need a discrete onehot-encoded object. Use the 'prep_data' function to prepare your data.**
    
    Input:  (i)   Zs: Pandas dataframe with discrete onehot-encoded Z (typically y in classification problems) from the source population (test set prefered) - *use the 'prep_data' function to prepare your data*;
            (ii)  Zt: Pandas dataframe with discrete onehot-encoded Z (typically y in  classification problems) from the target population (test set prefered) - *use the 'prep_data' function to prepare your data*;
            (iii) shift_model: KL model used to estimate the Dkl between Pt and Ps (trained using training set);
            (iv)  B: number of permutations used to calculate p-value;
            
    Output: (i) Dictionary containing the pvalue, the estimate of the shift (Dkl's) and the permutations values;


#### 2.3.4\. Function: `LocalPermut(Xs, ys, Xt, yt, totshift_model, labshift_model, task, n_bins=10, B=500, verbose=True)`
    
    Function that returns the local permutation p-values for testing H0 (Pt=Ps) for the conditional distributions of X|Y (y discrete)
    
    Input:  (i)   Xs and ys: Two Pandas dataframes with X and y from the source population (test set prefered) - use the 'prep_data' function to prepare your data;
            (ii)  Xt and yt: Two Pandas dataframes with X and y from the target population (test set prefered) - use the 'prep_data' function to prepare your data;
            (iii) totshift_model: KL model used to estimate the Dkl between the two joint distributions of (X,y) (trained using training set);
            (iv)  labshift_model: KL model used to estimate the Dkl between the two marginal distributions of labels y (trained using training set);
            (v)   task: 'class' or 'reg' for classification or regression;
            (vi)  n_bins: number of bins if performing regression task. If task=='reg', this function will evenly bin ys, yt based on y=(ys,yt) quantiles. We use binning only to get the p-value and we report the original KL estimate;
            (vii) B: number of permutations used to calculate p-value;
            
    Output: (i) Dictionary containing the pvalue, the estimate of the shift (DKL's) and the permutations values. In case of label binning, this function uses the binned variables to get the pvalue but it will return the non-binned DKL estimate;

#### 2.3.5\. Function: `CondRand(Xs, ys, Xt, yt, cd_model, totshift_model, covshift_model,B=500, verbose=True)`
    
    Function that returns the conditional randomization p-values for testing H0 (Pt=Ps) for the conditional distributions of Y|X
    
    Input:  (i)   Xs and ys: Two Pandas dataframes with X and y from the source population (test set prefered) - use the 'prep_data' function to prepare your data;
            (ii)  Xt and yt: Two Pandas dataframes with X and y from the target population (test set prefered) - use the 'prep_data' function to prepare your data;
            (iii) cd_model: conditional density model equiped with 'sample' function. See *cdist* module for more details;
            (iv)  totshift_model: KL model used to estimate the Dkl between the two joint distributions of (X,y) (trained using training set);
            (v)   covshift_model: KL model used to estimate the Dkl between the two marginal distributions of features X (trained using training set);
            (v)   B: number of permutations used to calculate p-value;
            
    Output: (i) Dictionary containing the pvalue, the estimate of the shift (Dkl's) and the permutations values;
