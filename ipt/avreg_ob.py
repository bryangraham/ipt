# Load library dependencies
import numpy as np
import numpy.linalg

import scipy as sp
import scipy.optimize
import scipy.stats

import pandas as pd

# Import additional functions called by eplm()
from .print_coef import print_coef
from .ols import ols
from .logit import logit
from .poisson import poisson
from .iv import iv


# Define avreg_dr function
#-----------------------------------------------------------------------------#
def avreg_ob(Y, X, W, c_id=None, s_wgt=None, silent=False):
    
    """
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu
    DATE: 9 July 2018    
    
    OVERVIEW
    --------
    This function computes the average partial effect of X on Y using the
    variance of the Oaxaca-Blinder regression estimator described in
    Graham and Pinto (2018). In this estimator the conditional linear
    predictor of Y given X, conditional on W, is assumed to equal
    E*[Y|X;W] = a0 + a1(W-mu_W) + b0 + b1(W-mu_W)X. See Assumption 6 of
    Graham and Pinto (2018).
        
    INPUTS:
    -------
    Y        : n X 1 pandas.Series of dependent variable
    X        : n X 1 pandas.Series of regressor entering linearly (i.e., policy variable)
               (currently program only accomodates a single "policy" variable)
    W        : n X J pandas.DataFrame of (functions of) control variables 
    c_id     : n X 1 pandas.Series of unique `cluster' id values (assumed to be integer valued) (optional)
               NOTE: Default is to assume independent observations and report heteroscedastic robust 
                     standard errors
    s_wgt    : n X 1 pandas.Series of sampling weights variable (optional)
    silent   : if set equal to True, then suppress all outcome (optional)
    
    NOTE     : Neither X nor W should include a constant. This is added below.
        
    OUTPUTS:
    --------
    beta_hat         : K x 1 vector of average coefficient estimates
    vcov_beta_hat    : K x K (cluster) Huber-robust variance-covariance estimate
    
    FUNCTIONS CALLED  : ...ols(), logit(), poisson()...
    ----------------    
    """
    
    #--------------------------------------------------------------------#
    #- STEP 1 : Organize data for average regression estimation         -#
    #--------------------------------------------------------------------#
    
    n       = len(Y)                     # Number of observations
    J       = W.shape[1]                 # Number of control variables
    K       = 1                          # Program only accomodates 1 policy variable/treatment
                                         # in its current implementation
    
    # Normalize weights to have mean one (if needed)
    if s_wgt is None:
        sw = 1 
    else:
        s_wgt_var = s_wgt.name                               # Get sample weight variable name
        sw = np.asarray(s_wgt/s_wgt.mean()).reshape((-1,1))  # Normalized sampling weights with mean one
    
    # Extract variable names from pandas data objects
    dep_var = Y.name                   # Get dependent variable names
    pol_var = X.name                   # Get policy variable name
    con_var = list(W.columns)          # Get control variable names
  
    #--------------------------------------------------------------------#
    #- STEP 2 : Construct regressor and instrument vectors              -#
    #--------------------------------------------------------------------#
    
    mu_W    = (sw * W).mean(axis=0)                                            # (Weighted) average of W
    Wc      = W.values.reshape((n,J)) - mu_W.values.reshape((1,J))             # n x J
    Wc_by_X = Wc * X.values.reshape((n,1))                                     # n x JK (K=1)
    
    # Regressor vector
    RX      = np.hstack((np.ones((n,1)), Wc, Wc_by_X, X.values.reshape((n,1)))) # n x 1 + J + JK + K (K=1)
    interactions = [control + "_X_" + pol_var for control in con_var]
    RX_names = ['constant'] + con_var + interactions + [pol_var]
    RX = pd.DataFrame(data = RX, columns = RX_names)
    
    #--------------------------------------------------------------------#
    #- STEP 3 : Compute average regression estimate of beta             -#
    #--------------------------------------------------------------------#
    [lambda_beta_hat, _, _, _, _] = ols(Y, RX, c_id=c_id, s_wgt=s_wgt, nocons=True, silent=True)
    
    # Extract coefficient sub-vectors    
    gamma_hat = lambda_beta_hat[1:(1+J),0].reshape((J,1))
    delta_hat = lambda_beta_hat[(1+J):(1+J+J*K),0].reshape((J*K,1))
    beta_hat  = lambda_beta_hat[(1+J+J*K):(1+J+J*K+K),0].reshape((K,1))
    
    #--------------------------------------------------------------------#
    #- STEP 4 : Compute variance-covariance matrix                      -#
    #--------------------------------------------------------------------#
    
    Y       = Y.values.reshape((n,1))                    # Turn pandas.Series into n x 1 numpy array
    RX      = RX.values.reshape((n,1+J+J*K+K))           # Turn pandas.DataFrame into n x 1 + J + JK + K numpy array
    X       = RX[:,(1+J+J*K):(1+J+J*K+K)].reshape((n,K)) # Get X submatrix/vector
    
    # Construct stacked moment vector
    m_1     = sw * Wc                              # n x J moment vector for estimating mean of W vector
    m_2     = sw * RX * (Y - RX @ lambda_beta_hat) # n x 1 + J + JK + K moment vector for estimating lambda and beta
    m       = np.hstack((m_1, m_2))                # n x J + 1 + J + JK + K matrix of full moments
    
    # Construct partitioned Jacobian matrix
   
    # (a) Form B matrix
    B   = (np.kron(np.eye(J), X) @ delta_hat).reshape((n,J), order='F')   # nJ x 1 vector reshaped into n x J matrix
    B   = ((sw * RX).T @ (gamma_hat.T + B))/n                             # Full B matrix, 1 + J +JK + K x J
    
    # (b) Put together full Jacobian matrix
    M1 = np.hstack((np.eye(J), np.zeros((J,1+J+J*K+K))))                  # First J rows of Jacobian matrix
    M2 = np.hstack((-B, ((sw * RX).T @ RX)/n))                            # Last 1+J+JK rows of Jacobian matrix
    M  = -np.vstack((M1, M2))                                             # Full Jacobian matrix 
    
    # (c) Compute covariance matrix of the scores
    if c_id is None: 
        
        # Compute variance-covariance matrix of moments (unclustered)
        fsc   = n/(n-J-1-J*K-K)                             # Finite-sample correction factor
        omega = fsc*(m.T @ m)/(n**2)                        # variance-covariance of the full moment vector
        
    else:
        
        # Get number and unique list of clusters
        c_list  = c_id.unique()            
        N       = len(c_list)    
        
        # Calculate cluster-robust variance-covariance matrix of moments
        # Sum score vector within clusters
        sum_m = np.empty((N,J+1+J+J*K+K))
    
        for c in range(0,N):
            b_cluster  = np.nonzero((c_id == c_list[c]))[0]                               # Indices of observations in c-th cluster 
            sum_m[c,:] = np.sum(m[np.ix_(b_cluster, range(0,J+1+J+J*K+K))], axis = 0)     # Sum over rows within c-th cluster
            # NOTE: Above line uses "open mesh" numpy indexing, see documentation
            
        # Compute variance-covariance matrix of summed moments (clustered)
        fsc   = (n/(n-J-1-J*K-K))*(N/(N-1))                 # Finite-sample correction factor
        omega = fsc*(sum_m.T @ sum_m)/(n**2)                # variance-covariance of the full summed moment vector
    
    # Put together full variance-covariance matrix
    iM = np.linalg.inv(M)
    vcov_theta_ehat = iM @ omega @ iM.T   
    vcov_beta_hat = vcov_theta_ehat[(J+1+J+J*K):(J+1+J+J*K+K),(J+1+J+J*K):(J+1+J+J*K+K)].reshape((K,K))             
    
    if not silent:
        print("")
        print("-----------------------------------------------------------------------")
        print("-     OAXACA-BLINDER AVERAGE REGRESSION ESTIMATION RESULTS            -")
        print("-----------------------------------------------------------------------")
        print("Dependent variable:        " + dep_var)
        print("Number of observations, n: " + "%0.0f" % n)
        print("")
        
        # Print coefficient estimate, standard error and 95 percent confidence interval
        print_coef(beta_hat, vcov_beta_hat, var_names=[pol_var], alpha=0.05)
       
        
        if c_id is None:
            print("NOTE: Heteroscedastic-robust standard errors reported")
        else:
            print("NOTE: Cluster-robust standard errors reported")
            print("      Cluster-variable   = " + c_id.name)
            print("      Number of clusters = " + "%0.0f" % N)
        if s_wgt is not None:
            print("NOTE: (Sampling) Weighted estimates computed.")
            print("      Weight-variable   = " + s_wgt_var)    
        
        print("-----------------------------------------------------------------------")
        print("- Control variables, W                                                -")
        print("-----------------------------------------------------------------------")
        for control in con_var:
            print(control.ljust(25))
        print("")
    
    return [beta_hat, vcov_beta_hat]
