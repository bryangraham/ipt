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


# Define avreg_ipw function
#-----------------------------------------------------------------------------#
def avreg_ipw(Y, X, W, psmodel='normal', c_id=None, s_wgt=None, silent=False):
    
    """
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu
    DATE: 9 July 2018    
    
    OVERVIEW
    --------
    This function computes the average partial effect of X on Y using the generalized
    "inverse probability weighting" estimator introduced by Wooldridge (2004).
    The large sample properties of this estimator are derived and discussed
    in Graham and Pinto (2018).
     
    INPUTS:
    -------
    Y        : n X 1 pandas.Series of dependent variable
    X        : n X 1 pandas.Series of regressor entering linearly (i.e., policy variable)
               (currently program only accomodates a single "policy" variable)
    W        : n X J pandas.DataFrame of (functions of) control variables 
    psmodel  : Model for f(X|W), 'normal', 'logit' or 'poisson'
    c_id     : n X 1 pandas.Series of unique `cluster' id values (assumed to be integer valued) (optional)
               NOTE: Default is to assume independent observations and report heteroscedastic robust 
                     standard errors
    s_wgt    : n X 1 pandas.Series of sampling weights variable (optional)
    silent   : if set equal to True, then suppress all outcome (optional)
    
    NOTE     : Neither X nor W should include a constant. This is added below.
        
    OUTPUTS:
    --------
    beta_hat          : K x 1 vector of average coefficient estimates
    vcov_beta_hat     : K x K (cluster) Huber-robust variance-covariance estimate
    
    FUNCTIONS CALLED  : ...ols(), logit(), poisson()...
    ----------------    
    """
    
    def normal_evhat(X, W, s_wgt=None, nocons=False):
        
        # Compute ehat and vhat by normal mle (i.e., ols)
        [pi_hat, _, H, S_i, ehat] = ols(X, W, c_id=None, s_wgt=s_wgt, nocons=nocons, silent=True)
        vhat = np.var(X - ehat.flatten(), ddof=np.shape(W)[1])
        vhat = np.full_like(ehat, vhat)
    
        return [ehat, vhat, H, S_i]

    def logit_evhat(X, W, s_wgt=None, nocons=False):
        
        # Compute ehat and vhat by logit mle
        [pi_hat, _, H, S_i, ehat, _] \
            = logit(X, W, s_wgt=s_wgt, nocons=nocons, c_id=None, silent=True, full=False)
        vhat = ehat * (1-ehat)    
        
        return [ehat, vhat, H, S_i]

    def poisson_evhat(X, W, s_wgt=None, nocons=False):
        
        # Compute ehat and vhat by poisson mle
        [pi_hat, _, H, S_i, ehat, _] \
            = poisson(X, W, c_id=None, s_wgt=s_wgt, nocons=nocons, silent=True, full=False, phi_sv=None)
        vhat = ehat
        
        return [ehat, vhat, H, S_i]

    #--------------------------------------------------------------------#
    #- STEP 1 : Compute estimate of e(W) = E[X|W] and v(W) = V(X|W)     -#
    #--------------------------------------------------------------------#
    
    psmodel_dict = {"normal": normal_evhat, "logit": logit_evhat, "poisson": poisson_evhat}
    [ehat, vhat, H, S_i] = psmodel_dict[psmodel](X, W, s_wgt=s_wgt, nocons=False)
    
    #--------------------------------------------------------------------#
    #- STEP 2 : Organize data for average regression estimation         -#
    #--------------------------------------------------------------------#
    
    n       = len(Y)                     # Number of observations
    J       = W.shape[1]                 # Number of control variables
    K       = 1                          # Program only accomodates 1 policy variable/treatment
                                         # in its current implementation
    L       = 1 + J                      # Number of parameters in propensity score model                                     
    
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
    #- STEP 3 : Compute average regression estimate of beta             -#
    #--------------------------------------------------------------------#
    
    # Calculate Wooldridge's (2004) (generalized) IPW estimator
    gw = sw*(X.values.reshape((n,1)) - ehat)/vhat
    beta_hat = np.sum(gw*Y.values.reshape((n,1)))/np.sum(gw*X.values.reshape((n,1))).reshape(-1,1)
    
    #--------------------------------------------------------------------#
    #- STEP 4 : Compute variance-covariance matrix                      -#
    #--------------------------------------------------------------------#
          
    # Compute influence function of estimator
    m = (gw*(Y.values.reshape((n,1)) - X.values.reshape((n,1)) @ beta_hat)).reshape(-1,1)
       
    # Compute PI_ms (divide by sw to get sample weighting right since S_i is already weighted)  
    ss  = (S_i/sw).T @ S_i
    sm  = (S_i/sw).T @ m 
    PI_ms = numpy.linalg.solve(ss, sm)
    
    # Influence function ("Residualized moment")
    m_tilde = m - S_i @ PI_ms
    
    # Compute covariance matrix of the scores
    if c_id is None: 
        
        # Compute variance-covariance matrix of moments (unclustered)
        fsc           = n/(n-L-K)                                   # Finite-sample correction factor
        vcov_beta_hat = fsc*(m_tilde.T @ m_tilde)/(n**2)            # variance-covariance of the full moment vector
        
    else:
        
        # Get number and unique list of clusters
        c_list  = c_id.unique()            
        N       = len(c_list)    
        
        # Calculate cluster-robust variance-covariance matrix of moments
        # Sum score vector within clusters
        sum_m = np.empty((N,K))
    
        for c in range(0,N):
            b_cluster  = np.nonzero((c_id == c_list[c]))[0]                        # Indices of observations in c-th cluster 
            sum_m[c,:] = np.sum(m_tilde[np.ix_(b_cluster, range(0,K))], axis = 0)  # Sum over rows within c-th cluster
            # NOTE: Above line uses "open mesh" numpy indexing, see documentation
            
        # Compute variance-covariance matrix of summed moments (clustered)
        fsc           = (n/(n-L-K))*(N/(N-1))             # Finite-sample correction factor
        vcov_beta_hat = fsc*(sum_m.T @ sum_m)/(N**2)      # variance-covariance of the full summed moment vector
        
    if not silent:
        print("")
        print("-----------------------------------------------------------------------")
        print("-   GENERALIZED IPW AVERAGE REGRESSION ESTIMATION RESULTS             -")
        print("-----------------------------------------------------------------------")
        print("Dependent variable:        " + dep_var)
        print("Number of observations, n: " + "%0.0f" % n)
        print("Propensity score model:    " + psmodel)
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
    
    