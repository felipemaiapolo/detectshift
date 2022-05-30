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
    2. [Step to use](#1.2)
2. [Modules](#2)
    1. [*tools*](#2.1)
    2. [*cdist*](#2.2)
    3. [*tests*](#2.3)


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

- **[Total Dataset Shift]:** $$H_{0,\text{D}}:P_{X,Y}=Q_{X,Y}$$
- **[Covariate Shift]:** $$H_{0,\text{C}}:P_{X}=Q_{X}$$
- **[Label Shift]:** $$H_{0,\text{L}}:P_{Y}=Q_{Y}$$ 
- **[Concept Shift 1]:** $$H_{0,\text{C1}}:P_{X|Y}=Q_{X|Y}$$
- **[Concept Shift 2]:** $$H_{0,\text{C2}}: P_{Y|X}=Q_{Y|X}$$ 

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

- Loading *DetectShift*;
- Preparing data with *prep_data* function;
- Instantiating and training models to estimate KL divergence using *KL* class;
- Instantiating and training models to estimate the conditional distirbution of $Y|X$ using *cdist* module (in case of testing for concept shift of type 2;
- Testing different types of shift using the *tests* module. 


--------------

<a name="2"></a>
## 2\. Modules

<a name="2.1"></a>
### 2.1\.  *tools*

#### 2.1.1\. `prep_data(Xs, ys, Xt, yt, test=.1, task=None, random_state=42)` 
    

    
   
    Function that gets data and prepare it to run the tests
    
    Input:  (i)   Xs and Xt: 2d-numpy array or Pandas Dataframe containing features from the source and target domain;
            (ii)  ys and yt: 1d-numpy array or 1-column Pandas Dataframe containing labels. If task=='class', then ys and yt must contain all labels [0,1,...,K-1], where K is the number of classes;
            (iii) test: fraction of the data going to the test set;
            (iv)  task: 'reg' or 'class' for regression or classification;
            (v)   random_state: seed used in the data splitting
            
    Output: Xs_train, Xs_test, ys_train, ys_test, Zs_train, Zs_test
            Xt_train, Xt_test, yt_train, yt_test, Zt_train, Zt_test
 

