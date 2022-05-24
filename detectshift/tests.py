import pandas as pd
import numpy as np
from tqdm import tqdm
from .utils import *

def Permut(Zs, Zt, shift_model, B=500, verbose=True):
    
    '''
    Function that returns the permutation p-values for testing H0 (Pt=Ps) for distributions of Z, where Z can be X, y, or (X,y)
    
    Input:  (i)   Zs: Pandas dataframe with Z from the source population (test set prefered) - use the 'prep_data' function to prepare your data;
            (ii)  Zt: Pandas dataframe with Z from the target population (test set prefered) - use the 'prep_data' function to prepare your data;
            (iii) shift_model: KL model used to estimate the Dkl between Pt and Ps (trained using training set);
            (iv)  B: number of permutations used to calculate p-value;
            
    Output: (i) Dictionary containing the pvalue, the estimate of the shift (Dkl's) and the permutations values;
    '''
    
    #Preparing
    Z = pd.concat([Zs, Zt], axis=0) 
    shift = shift_model.predict(Zt)
    perm = []

    #Getting weights (R-N deriv./density ratio) estimates
    ws = shift_model.predict_w(Zs)
    wt = shift_model.predict_w(Zt)
    w = np.vstack((ws.reshape(-1,1), wt.reshape(-1,1)))
    
    #Performing permutation
    for b in tqdm(range(B), disable=not(verbose)):
        shuffle = np.random.choice(range(Z.shape[0]), size=(Z.shape[0],), replace=False)
        indt = shuffle[Zs.shape[0]:]
        perm.append(np.mean(np.log(w[indt])))
    perm = np.array(perm)
    
    #Enforcing uniformity of p-values under H0 (adding a very small random number - we guarantee every statistic has a different value)
    s=10**-10
    perm=perm+np.random.normal(0,s,perm.shape[0])
    shift=shift+float(np.random.normal(0,s,1))
    
    #Preparing output
    out = {}
    out['pval'] = (1+np.sum(perm >= shift))/(B+1)
    out['kl'] = shift
    out['perm'] = perm
    
    return out

def PermutDiscrete(Zs, Zt, B=500, verbose=True):
    
    '''
    Function that returns the permutation p-values for testing H0 (Pt=Ps) for distributions of Z, where Z can be X, y, or (X,y)
    
    Input:  (i)   Zs: Pandas dataframe with onehot encoded discrete Z from the source population - use the 'prep_data' function to prepare your data;
            (ii)  Zt: Pandas dataframe with onehot encoded discrete Z from the target population - use the 'prep_data' function to prepare your data;
            (iii) B: number of permutations used to calculate p-value;
            
    Output: (i) Dictionary containing the pvalue, the estimate of the shift (Dkl's) and the permutations values;
    '''
    
    #Preparing
    Zs=get_classes(Zs)
    Zt=get_classes(Zt)
    Z = np.hstack((Zs, Zt))
    shift = KL_multinomial(Zs, Zt)
    perm = []

    #Performing permutation
    for b in tqdm(range(B), disable=not(verbose)):
        shuffle = np.random.choice(range(Z.shape[0]), size=(Z.shape[0],), replace=False)
        inds = shuffle[:Zs.shape[0]]
        indt = shuffle[Zs.shape[0]:]
        perm.append(KL_multinomial(Z[inds], Z[indt]))
    perm = np.array(perm)
    
    #Enforcing uniformity of p-values under H0 (adding a very small random number - we guarantee every statistic has a different value)
    s=10**-10
    perm=perm+np.random.normal(0,s,perm.shape[0])
    shift=shift +float(np.random.normal(0,s,1))
    
    #Preparing output
    out = {}
    out['pval'] = (1+np.sum(perm >= shift))/(B+1)
    out['kl'] = shift
    out['perm'] = perm
    
    return out

def LocalPermut(Xs, ys, Xt, yt, 
                totshift_model, labshift_model, task, n_bins=10,
                B=500, verbose=True):
    
    '''
    Function that returns the local permutation p-values for testing H0 (Pt=Ps) for the conditional distributions of X|Y (y discrete)
    
    Input:  (i)   Xs and ys: Two Pandas dataframes with X and y from the source population (test set prefered) - use the 'prep_data' function to prepare your data;
            (ii)  Xt and yt: Two Pandas dataframes with X and y from the target population (test set prefered) - use the 'prep_data' function to prepare your data;
            (iii) totshift_model: KL model used to estimate the Dkl between the two joint distributions of (X,y) (trained using training set);
            (iv)  labshift_model: KL model used to estimate the Dkl between the two marginal distributions of labels y (trained using training set);
            (v)   task: 'class' or 'reg' for classification or regression;
            (vi)  n_bins: number of bins if performing regression task. If task=='reg', this function will evenly bin ys, yt based on y=(ys,yt) quantiles;
            (vii) B: number of permutations used to calculate p-value;
            
    Output: (i) Dictionary containing the pvalue, the estimate of the shift (DKL's) and the permutations values. In case of label binning, this function uses the binned variables to get the pvalue but it will return the non-binned DKL estimate;
    '''
    
    ##Checking task
    if task not in ['reg','class']:
        raise ValueError("'task' must be in ['reg','class'].")
    
    #Storing columns names
    X_names=list(Xs.columns)
    y_names=list(ys.columns)
    
    #Creating auxiliary ys2, yt2, and y2
    if task=='class':
        ys2=get_classes(ys)
        yt2=get_classes(yt)
    else:
        #cut-points
        perc=list(range(int(100/n_bins),100,int(100/n_bins)))
        ybins = np.percentile(np.hstack((np.array(ys).squeeze(),np.array(yt).squeeze())), q=perc)
        #values for each bin
        perc_values = list(range(int(100/(2*n_bins)),100,int(100/(2*n_bins))))[::2]
        ybins_values = np.percentile(np.hstack((np.array(ys).squeeze(),np.array(yt).squeeze())), q=perc_values)
        ys2 = np.array([ybins_values[i] for i in np.digitize(np.array(ys), ybins)])
        yt2 = np.array([ybins_values[i] for i in np.digitize(np.array(yt), ybins)])
        
    y2=np.hstack(((ys2.squeeze(),yt2.squeeze())))
    
    #Estimating Total Shift (KL_{X,Y})
    if task=='class':
        Zt_new = pd.concat([Xt.reset_index(drop=True), yt.reset_index(drop=True)], axis=1)
    else:
        Zt_new = pd.concat([Xt.reset_index(drop=True), pd.DataFrame(yt2).reset_index(drop=True)], axis=1)
    Zt_new.columns = list(X_names)+list(y_names) 
    totshift  = totshift_model.predict(Zt_new)
    
    #Estimating Label Shift (KL_Y)
    if labshift_model==None:
        if task not in ['class']: raise ValueError("If labshift_model==None, then task must be 'class'.")
        labshift = KL_multinomial(ys2, yt2)
    else:
        labshift  = labshift_model.predict(yt)
    
    #Estimating Concept Shift 1 (KL_{X|Y})
    concshift = totshift-labshift
    
    #Recording possible values for y2
    Y=np.unique(y2)
    ind={}
    for j in Y:
        ind[j] = (y2==j).squeeze()
    
    #Performing local permutation
    X = pd.concat([Xs, Xt], axis=0) 
    totperm = []
    
    for b in tqdm(range(B), disable=not(verbose)):
        Xt_perm = pd.DataFrame(np.zeros(Xt.shape))
        shuffle={}
        indt={}
        for j in Y:
            shuffle[j] = np.random.choice(range(np.sum(ind[j])), size=(np.sum(ind[j]),), replace=False)
            indt[j] = shuffle[j][:np.sum(yt2==j)]
            ind_perm=[i for i, x in enumerate(yt2.squeeze()==j) if x] #getting positions where True
            Xt_perm.iloc[ind_perm,:] = X.loc[ind[j],:].iloc[indt[j],:] #ind_perm is a list that given indices s.t. yt==j; then we are randomly assigning new values to Xt_perm.iloc[ind_perm,:] from X.loc[ind[j],:].iloc[indt[j],:], where X.loc[ind[j],:] selects all rows of X=(Xs,Xt) s.t. y==j and then randomly select from those rows a subset of size np.sum(np.array(yt)==j) (that's where the permutation comes from).
        if task=='class':
            Zt_new = pd.concat([Xt_perm.reset_index(drop=True), yt.reset_index(drop=True)], axis=1)
        else:
            Zt_new = pd.concat([Xt_perm.reset_index(drop=True), pd.DataFrame(yt2).reset_index(drop=True)], axis=1)
        Zt_new.columns = list(X_names)+list(y_names) 
        totperm.append(totshift_model.predict(Zt_new))
        
    perm = (np.array(totperm) - np.array(labshift)).tolist()
    perm = np.array(perm)
    
    #Enforcing uniformity of p-values under H0 (adding a very small random number - we guarantee every statistic has a different value)
    s=10**-10
    perm=perm+np.random.normal(0,s,perm.shape[0])
    concshift=concshift+float(np.random.normal(0,s,1))
    
    #Preparing output
    out = {}
    out['pval'] = (1+np.sum(perm >= concshift))/(B+1)
    
    #Updating Conc. Shift if  we discretized Y to run the test
    if task=='reg':
        Zt_new = pd.concat([Xt.reset_index(drop=True), yt.reset_index(drop=True)], axis=1)
        Zt_new.columns = list(X_names)+list(y_names) 
        totshift  = totshift_model.predict(Zt_new)
        concshift = totshift-labshift
    
    out['kl'] = concshift
    out['perm'] = perm
    
    return out

def CondRand(Xs, ys, Xt, yt, 
             cd_model, totshift_model, covshift_model,
             B=500, verbose=True):
    
    '''
    Function that returns the conditional randomization p-values for testing H0 (Pt=Ps) for the conditional distributions of Y|X
    
    Input:  (i)   Xs and ys: Two Pandas dataframes with X and y from the source population (test set prefered) - use the 'prep_data' function to prepare your data;
            (ii)  Xt and yt: Two Pandas dataframes with X and y from the target population (test set prefered) - use the 'prep_data' function to prepare your data;
            (iii) cd_model: conditional density model equiped with 'sample' function. See documentation for more details;
            (iv)  totshift_model: KL model used to estimate the Dkl between the two joint distributions of (X,y) (trained using training set);
            (v)   covshift_model: KL model used to estimate the Dkl between the two marginal distributions of features X (trained using training set);
            (v)   B: number of permutations used to calculate p-value;
            
    Output: (i) Dictionary containing the pvalue, the estimate of the shift (Dkl's) and the permutations values;
    '''
    
    #Storing columns names
    X_names=list(Xs.columns)
    y_names=list(ys.columns)

    #Preparing data
    Zs = pd.concat([Xs.reset_index(drop=True), ys.reset_index(drop=True)], axis=1) 
    Zt = pd.concat([Xt.reset_index(drop=True), yt.reset_index(drop=True)], axis=1) 
            
    #Estimating shifts (KLs)
    covshift  = covshift_model.predict(Xt) 
    totshift  = totshift_model.predict(Zt)
    concshift = totshift-covshift
    
    #Performing conditional randomization
    totperm = []
    for b in tqdm(range(B), disable=not(verbose)):   
        yt = cd_model.sample(Xt)    
        Zt_new = pd.concat([Xt.reset_index(drop=True), yt.reset_index(drop=True)], axis=1)  
        Zt_new.columns = list(X_names)+list(y_names) 
        totperm.append(totshift_model.predict(Zt_new))
        
    perm = (np.array(totperm) - np.array(covshift)).tolist()
    perm = np.array(perm)
    
    #Enforcing uniformity of p-values under H0 (adding a very small random number - we guarantee every statistic has a different value)
    s=10**-10
    perm=perm+np.random.normal(0,s,perm.shape[0])
    concshift=concshift+float(np.random.normal(0,s,1))
    
    #Preparing output
    out = {}
    out['pval'] = (1+np.sum(perm >= concshift))/(B+1)
    out['kl'] = concshift
    out['perm'] = perm
    
    return out

def ShiftDiagnostics(Xs_test, ys_test, Xt_test, yt_test,
                     totshift_model, covshift_model, labshift_model,
                     cd_model, task, n_bins=10, B=500,
                     verbose=True):
    '''
    Function that returns results for all the tests
    
    Input:  (i)    Xs_test and ys_test: Two Pandas dataframes with X and y from the source population - use the 'prep_data' function to prepare your data;
            (ii)   Xt_test and yt_test: Two Pandas dataframes with X and y from the target population - use the 'prep_data' function to prepare your data;
            (iii)  totshift_model: KL model used to estimate the Dkl between the two joint distributions of (X,y) (trained using training set);
            (iv)   covshift_model: KL model used to estimate the Dkl between the two marginal distributions of features X (trained using training set);
            (v)    labshift_model: KL model used to estimate the Dkl between the two marginal distributions of labels y (trained using training set) - you can set labshift_model=None if task=='class' and, in this case, the function will call "KL_multinomial" as estimator;
            (vi)   cd_model: conditional density model equiped with 'sample' function. See documentation for more details;
            (vii)  task: 'class' or 'reg' for classification or regression;
            (viii) n_bins: number of bins if performing regression task. If task=='reg', this function will evenly bin ys, yt based on y=(ys,yt) quantiles;
            (ix)   B: number of permutations used to calculate p-value;
            
    Output: (i) Dictionary containing the pvalues, the estimates of the shifts (Dkl's) and the permutations values;
    '''
        
    ##Checking task
    if task not in ['reg','class']:
        raise ValueError("'task' must be in ['reg','class'].")
        
    if verbose: print("Calculating p-value for total shift...") 
    Zs_test = pd.concat([Xs_test.reset_index(drop=True), ys_test.reset_index(drop=True)], axis=1)  
    Zt_test = pd.concat([Xt_test.reset_index(drop=True), yt_test.reset_index(drop=True)], axis=1) 
    tot=Permut(Zs_test, Zt_test, totshift_model, B=B, verbose = verbose)
    
    if verbose: print("Calculating p-value for label shift...")
    if labshift_model==None: 
        lab=PermutDiscrete(ys_test, yt_test, B=B, verbose = verbose)
    else: 
        lab=Permut(ys_test, yt_test, labshift_model, B=B, verbose = verbose)
    
    if verbose: print("Calculating p-value for covariate shift...")
    cov=Permut(Xs_test, Xt_test, covshift_model, B=B, verbose = verbose)
    
    if verbose: print("Calculating p-value for concept shift type 1...")
    conc1=LocalPermut(Xs_test, ys_test, Xt_test, yt_test, 
                      totshift_model, labshift_model, task, n_bins, B=B, verbose = verbose)
    
    if verbose: print("Calculating p-value for concept shift type 2...")
    conc2=CondRand(Xs_test, ys_test, Xt_test, yt_test, 
                   cd_model, totshift_model, covshift_model, B=B, verbose = verbose)
    
    return {'tot':tot, 'lab':lab, 'cov':cov, 'conc1':conc1, 'conc2':conc2}
