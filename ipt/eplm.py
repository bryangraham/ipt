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


# Define eplm() function
#-----------------------------------------------------------------------------#
def eplm(Y, X, W, psmodel='normal', c_id=None, s_wgt=None, nocons=False, silent=False):
    
    """
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu
    DATE: 9 July 2018    
    
    OVERVIEW
    --------
    This function computes E-estimates of the coefficient on the scalar variable X (K=1)
    which enters linearly in a partially linear model. The estimator is a variant of one 
    due to Newey (1990, JAE) and Robins, Mark and Newey (1992, Biometrics). It is described 
    by Graham (2018), who characterizes its local semiparametric efficiency and double
    robustness properties.
    
    INPUTS:
    -------
    Y        : n X 1 pandas.Series of dependent variable
    X        : n X 1 pandas.Series of regressor entering linearly (i.e., policy variable)
               (currently program only accomodates a single "policy" variable)
    W        : n X J pandas.DataFrame of (functions of) control variables
    psmodel  : Model for e(W) = E[X|W], 'normal', 'logit' or 'poisson'
    c_id     : n X 1 pandas.Series of unique `cluster' id values (assumed to be integer valued) (optional)
               NOTE: Default is to assume independent observations and report heteroscedastic robust 
                     standard errors
    s_wgt    : n X 1 pandas.Series of sampling weights variable (optional)
    nocons   : If True, then do NOT add constant to W matrix
                (only set to True if user passes in a dataframe with a constant included)
    silent   : if set equal to True, then suppress all outcome (optional)
        
    OUTPUTS:
    --------
    beta_ehat          : K x 1 vector of ML estimates of phi = (kappa, gamma')'
    vcov_beta_ehat     : K x K (cluster) Huber-robust variance-covariance estimate
    
    FUNCTIONS CALLED  : ...ols(), logit(), poisson()...
    ----------------    
    """
    
    def normal_ehat(X, W, s_wgt=None, nocons=False):
        
        # Compute ehat by normal mle (i.e., ols)
        [pi_hat, _, H, S_i, ehat] = ols(X, W, c_id=None, s_wgt=s_wgt, nocons=nocons, silent=True)
    
        return [ehat, H, S_i]

    def logit_ehat(X, W, s_wgt=None, nocons=False):
        
        # Compute ehat by logit mle
        [pi_hat, _, H, S_i, ehat, _] \
            = logit(X, W, s_wgt=s_wgt, nocons=nocons, c_id=None, silent=True, full=False)
        
        return [ehat, H, S_i]

    def poisson_ehat(X, W, s_wgt=None, nocons=False):
        
        # Compute ehat by poisson mle
        [pi_hat, _, H, S_i, ehat, _] \
            = poisson(X, W, c_id=None, s_wgt=s_wgt, nocons=nocons, silent=True, full=False, phi_sv=None)
        
        return [ehat, H, S_i]

    #--------------------------------------------------------------------#
    #- STEP 1 : Compute estimate of e(W) = E[X|W]                       -#
    #--------------------------------------------------------------------#
    
    psmodel_dict = {"normal": normal_ehat, "logit": logit_ehat, "poisson": poisson_ehat}
    [ehat, H, S_i] = psmodel_dict[psmodel](X, W, s_wgt=s_wgt, nocons=nocons)
    
    #--------------------------------------------------------------------#
    #- STEP 2 : Organize data for E-estimation                          -#
    #--------------------------------------------------------------------#
    
    n       = len(Y)                     # Number of observations
    J       = W.shape[1]                 # Number of control variables
    K       = 1                          # Program only accomodates 1 policy variable/treatment
                                         # in its current form
    
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
    #- STEP 3 : Compute E-estimate of beta                              -#
    #--------------------------------------------------------------------#
    
    X = pd.DataFrame(data = X, columns=[pol_var])
    Z = pd.DataFrame(data = X[pol_var].values.reshape((n,1)) - ehat, columns=["X - e(W)"])
    
    [beta_ehat, vcov_beta_hat, _] = iv(Y, X, Z, c_id=c_id, s_wgt=s_wgt, nocons=True, silent=True)
    
    #--------------------------------------------------------------------#
    #- STEP 4 : Compute variance-covariance matrix                      -#
    #--------------------------------------------------------------------#
    
    Y       = Y.values.reshape((n,1))  # Turn pandas.Series into n x 1 numpy array
    X       = X.values.reshape((n,K))  # Turn pandas.DataFrame into n x K numpy array
    Z       = Z.values.reshape((n,K))  # Turn pandas.DataFrame into n x J numpy array
    
    m_i       = sw * Z * (Y - X @ beta_ehat)   # Estimating moment
    SS        = S_i.T @ S_i                    # Compute projection of moment on to e(W) scores
    Sm        = S_i.T @ m_i
    pi_hat    = np.linalg.solve(SS, Sm)        # J x 1 vector of projection coefficients
    m_i_tilde = m_i - S_i @ pi_hat             # n x 1 vector of residualized moments
                                               # (accounts for two-step estimation)
    
    if c_id is None: 
        
        # Compute variance-covariance matrix of scores (unclustered)
        fsc   = n/(n-K-J)                                   # Finite-sample correction factor
        omega = fsc*(m_i_tilde.T @ m_i_tilde)               # K X K variance-covariance of the residualized moments
        
    else:
        
        # Get number and unique list of clusters
        c_list  = c_id.unique()            
        N       = len(c_list)    
        
        # Calculate cluster-robust variance-covariance matrix of score_i
        # Sum score vector within clusters
        sum_m_tilde = np.empty((N,K))
    
        for c in range(0,N):
            b_cluster        = np.nonzero((c_id == c_list[c]))[0]                           # Indices of observations in c-th cluster 
            sum_m_tilde[c,:] = np.sum(m_i_tilde[np.ix_(b_cluster, range(0,K))], axis = 0)   # Sum over rows within c-th cluster
            # NOTE: Above line uses "open mesh" numpy indexing, see documentation
            
        # Compute variance-covariance matrix of residualized moments (clustered)
        fsc   = (n/(n-K-J))*(N/(N-1))                       # Finite-sample correction factor
        omega = fsc*(sum_m_tilde.T @ sum_m_tilde)           # K X K variance-covariance of the summed residualized moments
    
    # Compute variance-covariance matrix of phi_ML
    v_W  = (sw * Z).T @ X                                   # Expected conditional variance of the policy variable
    iv_W = np.linalg.inv(v_W)
    vcov_beta_ehat = iv_W @ omega @ iv_W.T                  
    
    if not silent:
        print("")
        print("-----------------------------------------------------------------------")
        print("-                    E-ESTIMATION RESULTS                             -")
        print("-----------------------------------------------------------------------")
        print("Dependent variable:        " + dep_var)
        print("Number of observations, n: " + "%0.0f" % n)
        print("Propensity score model:    " + psmodel)
        print("")
        
        # Print coefficient estimate, standard error and 95 percent confidence interval
        print_coef(beta_ehat, vcov_beta_ehat, var_names=[pol_var], alpha=0.05)
       
        
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
    
    return [beta_ehat, vcov_beta_ehat]
