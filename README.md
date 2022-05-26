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
1. [Introduction / Installing package](#1)
2. [Functions ](#2)

--------------

<a name="0"></a>
## 0\. Quick start (demo)

Below are the links to some demonstrations on how to use *DetectShift* in practice:

- **[Binary classification:](https://colab.research.google.com/github/felipemaiapolo/detectshift/blob/main/Classification2.ipynb)** In this notebook, we showcase an use example of dataset shift diagnostics when the response variable $Y$ is binary.

- **[Multinomial classification:](https://colab.research.google.com/github/felipemaiapolo/detectshift/blob/main/Classification2.ipynb)** In this notebook, we showcase an use example of dataset shift diagnostics when the response variable $Y$ is discrete with more than 2 values.

- **[Regression:](https://colab.research.google.com/github/felipemaiapolo/detectshift/blob/main/Regression1.ipynb)** In this notebook, we showcase an use example of dataset shift diagnostics when the response variable $Y$ is continuous.

- **[Regression with categorical features:](https://colab.research.google.com/github/felipemaiapolo/detectshift/blob/main/Regression2.ipynb)** In this notebook, we showcase an use example of dataset shift diagnostics when the response variable $Y$ is quantitative but discrete (with many different values). Also, in this example, we make use of categorical features, exploiting [Catboost](https://catboost.ai/) functionality.




--------------

<a name="1"></a>
## 1\. Introduction / 
*DetectShift* aims to quantify and test which types of dataset shift occur in a dataset. 

If (1) denotes the source distribution and (2) denotes the target distribution, the null hypotheses we want to test are:

$$H_0:P_{X,Y}=Q_{X,Y}$$

$$\beta_{i }(x) =$$

- **[Total Dataset Shift]:** $H_0:\mathcal{P}^{(1)}_{X,Y}=\mathcal{P}^{(2)}_{X,Y}$ 
- **[Covariate Shift]:** $H_{0,\text{C}}:P_{X}=Q_{X}$ 
- **[Label Shift]:** $H_{0,\text{L}}:\mathcal{P}^{(1)}_{Y}=\mathcal{P}^{(2)}_{Y}$ 
- **[Concept Shift 1]:** $H_{0,\text{C1}}:\mathcal{P}^{(1)}_{X|Y}=\mathcal{P}^{(2)}_{X|Y}$
- **[Concept Shift 2]:** $H_{0,\text{C2}}: \mathcal{P}^{(1)}_{Y|X}=\mathcal{P}^{(2)}_{Y|X}$ 

The test statistic used is a KL divergence estimator while the p-values are obtained through simulation (permutation/randomization) tests.

--------------

<a name="2"></a>
## 2\. Installing package / 


You can install our package running the following command on terminal
``` :sh
$ pip install git+https://github.com/felipemaiapolo/detectshift
```

You can load *DetectShift* in the following way

```python
import detectshift as ds
```
