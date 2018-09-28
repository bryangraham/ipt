# Load library dependencies
import numpy as np
import numpy.linalg

import pandas as pd

# Import print_coef() command in ipt module since ols() calls it
from .print_coef import print_coef

# Define ols() function
#-----------------------------------------------------------------------------#
def ols(Y, X, c_id=None, s_wgt=None, nocons=False, silent=False):
    """
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu  
    DATE: 26 May 2016, updated for Python 3.6 on 12 June 2018  
    
    This function returns OLS coefficient estimates associated with the
    linear regression fit of Y on X. It reports either heteroscedastic- or
    cluster-robust standard errors as directed by the user. The program
    also allows for the incorporation of sampling weights. While this
    function provides less features that Statsmodels implementation of
    OLS, it is designed to provide easy access to the handful of features
    needed most frequently for cross-section econometric analysis. The
    dependency on Pandas is introduced to provide a convenient way to
    include variable names, which are incorporated into the estimation
    output.    
    
    
    INPUTS:
    -------
    Y        : n X 1 pandas.Series of dependent variable
    X        : n X K pandas.DataFrame of regressors (see nocons parameter below)
    c_id     : n X 1 pandas.Series of unique `cluster' id values (assumed to be integer valued) (optional)
               NOTE: Default is to assume independent observations and report heteroscedastic robust 
                     standard errors
    s_wgt    : n X 1 pandas.Series of sampling weights variable (optional)
    nocons   : If True, then do NOT add constant to X matrix
               (only set to True if user passes in a dataframe with a constant included)
    silent   : if set equal to True, then suppress all outcome (optional)
    
    OUTPUTS:
    --------
    beta_hat      : K x 1 vector of linear ols estimates of beta
    vcov_beta_hat : K x K cluster-robust variance-covariance estimate
    hess_logl     : Hessian matrix associated with Gaussian log-likelihood
    score_i       : n x K matrix of likelihood score contributions for each unit
    ehat          : n x 1 vector of E[Y|X] fitted values

    FUNCTIONS CALLED : ...print_coef()...
    ----------------
    
    """
    n       = len(Y)                     # Number of observations
    K       = X.shape[1]                 # Number of regressors
    
    # Normalize weights to have mean one (if needed)
    if s_wgt is None:
        sw = 1 
    else:
        s_wgt_var = s_wgt.name                               # Get sample weight variable name
        sw = np.asarray(s_wgt/s_wgt.mean()).reshape((-1,1))  # Normalized sampling weights with mean one
    
    # Extract variable names from pandas data objects
    dep_var = Y.name                   # Get dependent variable names
    ind_var = list(X.columns)          # Get independent variable names
    
    # Transform pandas objects into appropriately sized numpy arrays
    Y       = Y.values.reshape((n,1))  # Turn pandas.Series into n x 1 numpy array
    X       = X.values                 # Turn pandas.DataFrame into n x K numpy array
    
    # Add a constant to the regressor matrix (if needed)
    if not nocons:
        X       = np.concatenate((np.ones((n,1)), X), axis=1) 
        ind_var = ['constant'] + ind_var
        K      += 1
    
    # Compute beta_hat   
    XX  = (sw * X).T @ X
    XY  = (sw * X).T @ Y
    beta_hat = np.linalg.solve(XX, XY)
    ehat    = X @ beta_hat
    
    # Compute estimate of variance-covariance matrix of the sample moment vector
    score_i   = sw * X * (Y - ehat)       # n x K matrix of moment/score vectors
    hess_logl = XX                        # K x K "hessian" of Gaussian log-likelihood
    
    if c_id is None: 
        
        # Calculate heteroscedastic robust variance-covariance matrix of psi
        fsc   = n/(n-K)                   # Finite-sample correction factor
        omega = fsc*(score_i.T @ score_i) # K X K variance-covariance of the moments
        
        iXX           = np.linalg.inv(XX)
        vcov_beta_hat = iXX @ omega @ iXX.T
        
    else:
        
        # Get number and unique list of clusters
        c_list  = np.unique(c_id)            
        N       = len(c_list)    
        
        # Calculate cluster-robust variance-covariance matrix of psi
        # Sum moment vector within clusters
        sum_score_i = np.empty((N,K))
    
        for c in range(0,N):
           
            b_cluster    = np.nonzero((c_id == c_list[c]))[0]                           # Observations in c-th cluster 
            sum_score_i[c,:] = np.sum(score_i[np.ix_(b_cluster, range(0,K))], axis = 0) # Sum over rows within c-th cluster
            
        # Compute variance-covariance matrix of beta_hat
        fsc   = (n/(n-K))*(N/(N-1))                      # Finite-sample correction factor
        omega = fsc*sum_score_i.T @ sum_score_i          # K X K variance-covariance of the summed moments
        
        iXX           = np.linalg.inv(XX)
        vcov_beta_hat = iXX.dot(omega).dot(iXX.T)                
 
    if not silent:
        
        
        
        
        print("")
        print("-----------------------------------------------------------------------")
        print("-                     OLS ESTIMATION RESULTS                          -")
        print("-----------------------------------------------------------------------")
        print("Dependent variable:        " + dep_var)
        print("Number of observations, n: " + "%0.0f" % n)
        print("")
        print("")
        
        # Print coefficient estimates, standard errors and 95 percent confidence intervals
        print_coef(beta_hat, vcov_beta_hat, var_names=ind_var, alpha=0.05)
        
        if c_id is None:
            print("NOTE: Heteroscedastic-robust standard errors reported")
        else:
            print("NOTE: Cluster-robust standard errors reported")
            print("      Cluster-variable   = " + c_id.name)
            print("      Number of clusters = " + "%0.0f" % N)
        if s_wgt is not None:
            print("NOTE: (Sampling) Weighted estimates computed.")
            print("      Weight-variable   = " + s_wgt_var)    

    return [beta_hat, vcov_beta_hat, hess_logl, score_i, ehat]