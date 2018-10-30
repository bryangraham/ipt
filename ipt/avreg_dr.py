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
def avreg_dr(Y, X, W, psmodel='normal', c_id=None, s_wgt=None, silent=False):
    
    """
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu
    DATE: 9 July 2018    
    
    OVERVIEW
    --------
    This function computes the average partial effect of X on Y using the
    locally efficient, doubly robust average regression estimator 
    introduced by Graham and Pinto (2018). This function currently restricts 
    X to be a scalar policy variable with a conditional distribution given W 
    that is normal, poisson or logit; W enters the canonical link function
    linearly and the intercept and slope coefficients are modelled "as if"
    E[A|W] and E[B|W] are both linear in W.
    
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
    #- STEP 3 : Construct regressor and instrument vectors              -#
    #--------------------------------------------------------------------#
    
    mu_W    = (sw * W).mean(axis=0)                                            # (Weighted) average of W
    Wc      = W.values.reshape((n,J)) - mu_W.values.reshape((1,J))             # n x J
    Wc_by_X = Wc * X.values.reshape((n,1))                                     # n x JK (K=1)
    
    # Regressor vector
    RX      = np.hstack((np.ones((n,1)), Wc, Wc_by_X, X.values.reshape((n,1)))) # n x 1 + J + JK + K (K=1)
    interactions = [control + "_X_" + pol_var for control in con_var]
    RX_names = ['constant'] + con_var + interactions + [pol_var]
    RX = pd.DataFrame(data = RX, columns = RX_names)
    
    # Instrument vector
    Z       = np.hstack((np.ones((n,1)), Wc, Wc_by_X, (X.values.reshape((n,1)) - ehat)/vhat)) # n x 1 + J + JK + K (K=1)
    Z_names = ['constant'] + con_var + interactions + ["(X - e(W))/v(W)"]    
    Z = pd.DataFrame(data = Z, columns = Z_names)
    
    #--------------------------------------------------------------------#
    #- STEP 4 : Compute average regression estimate of beta             -#
    #--------------------------------------------------------------------#
    
    [lambda_beta_hat, _, _] = iv(Y, RX, Z, c_id=c_id, s_wgt=s_wgt, nocons=True, silent=True)
    
    # Extract coefficient sub-vectors    
    alpha_hat = lambda_beta_hat[0,0].reshape((1,1))
    gamma_hat = lambda_beta_hat[1:(1+J),0].reshape((J,1))
    delta_hat = lambda_beta_hat[(1+J):(1+J+J*K),0].reshape((J*K,1))
    beta_hat  = lambda_beta_hat[(1+J+J*K):(1+J+J*K+K),0].reshape((K,1))
    
    #--------------------------------------------------------------------#
    #- STEP 5 : Compute variance-covariance matrix                      -#
    #--------------------------------------------------------------------#
    
    Y       = Y.values.reshape((n,1))                    # Turn pandas.Series into n x 1 numpy array
    RX      = RX.values.reshape((n,1+J+J*K+K))           # Turn pandas.DataFrame into n x 1 + J + JK + K numpy array
    R       = RX[:,0:(1+J+J*K)].reshape((n,1+J+J*K))     # Get R submatrix/vector
    X       = RX[:,(1+J+J*K):(1+J+J*K+K)].reshape((n,K)) # Get X submatrix/vector
    Z       = Z.values.reshape((n,1+J+J*K+K))            # Turn pandas.DataFrame into n x 1 + J + JK + K numpy array
    
    # Construct stacked moment vector
    m_1     = S_i                                 # n x L matrix of scores from generalized propensity score estimation
    m_2     = sw * Wc                             # n x J moment vector for estimating mean of W vector
    m_3     = sw * Z * (Y - RX @ lambda_beta_hat) # n x 1 + J + JK + K moment vector for estimating lambda and beta
    m       = np.hstack((m_1, m_2, m_3))          # n x L + J + 1 + J + JK + K matrix of full moments
    
    # Construct partitioned Jacobian matrix
    
    # (a) Form B1 matrix
    U_star = (Y - RX @ lambda_beta_hat)
    B1a = (np.kron(np.eye(J), X) @ delta_hat).reshape((n,J), order='F')   # nJ x 1 vector reshaped into n x J matrix
    B1b = ((sw * R).T @ (gamma_hat.T + B1a))/n                            # 1 + J + JK x J matrix  
    B1c = np.mean(sw * X * U_star, axis = 0).reshape((K,1))               # K x 1 vector estimating E[XU*]
    B1c = np.kron(np.eye(J), B1c)                                         # JK x J matrix
    B1  = B1b - np.vstack((np.zeros((1+J,J)), B1c))                       # Full B1 matrix, 1 + J +JK x J

    # (b) Form B2 matrix
    B2 = ((sw * Z[:,(1+J+J*K):(1+J+J*K+K)].reshape((n,K))).T @ B1a)/n     # K x J matrix
    
    # (c) Form lower-left K x L block of Jacobian
    LL = ((Z[:,(1+J+J*K):(1+J+J*K+K)] * U_star).reshape((n,K)).T @ S_i)/n # K x L matrix (already weighted via S_i)
    
    # (d) Put together full Jacobian matrix
    Ma = np.hstack((-H/n, np.zeros((L,J+1+J+J*K+K))))                                    # First L rows of Jacobian matrix
    Mb = np.hstack((np.zeros((J,L)), np.eye(J), np.zeros((J,1+J+J*K+K))))                # Next J rows of Jacobian matrix
    Mc = np.hstack((np.zeros((1+J+J*K,L)), -B1, ((sw * R).T @ R)/n, ((sw * R).T @ X)/n)) # Next 1+J+JK rows of Jacobian matrix
    Md = np.hstack((LL, -B2, np.zeros((K,1+J+J*K)), np.eye(K)))                          # Final K rows of the Jacobian matrix
    M  = -np.vstack((Ma, Mb, Mc, Md))                                                    # Full Jacobian matrix 
    
    # (e) Compute covariance matrix of the scores
    if c_id is None: 
        
        # Compute variance-covariance matrix of moments (unclustered)
        fsc   = n/(n-L-J-1-J*K-K)                           # Finite-sample correction factor
        omega = fsc*(m.T @ m)/(n**2)                        # variance-covariance of the full moment vector
        
    else:
        
        # Get number and unique list of clusters
        c_list  = c_id.unique()            
        N       = len(c_list)    
        
        # Calculate cluster-robust variance-covariance matrix of moments
        # Sum score vector within clusters
        sum_m = np.empty((N,L+J+1+J+J*K+K))
    
        for c in range(0,N):
            b_cluster  = np.nonzero((c_id == c_list[c]))[0]                               # Indices of observations in c-th cluster 
            sum_m[c,:] = np.sum(m[np.ix_(b_cluster, range(0,L+J+1+J+J*K+K))], axis = 0)   # Sum over rows within c-th cluster
            # NOTE: Above line uses "open mesh" numpy indexing, see documentation
            
        # Compute variance-covariance matrix of summed moments (clustered)
        fsc   = (n/(n-L-J-1-J*K-K))*(N/(N-1))               # Finite-sample correction factor
        omega = fsc*(sum_m.T @ sum_m)/(n**2)                # variance-covariance of the full summed moment vector
    
    # Put together full variance-covariance matrix
    iM = np.linalg.inv(M)
    vcov_theta_hat = iM @ omega @ iM.T   
    vcov_beta_hat = vcov_theta_hat[(L+J+1+J+J*K):(L+J+1+J+J*K+K),(L+J+1+J+J*K):(L+J+1+J+J*K+K)].reshape((K,K))             
    
    if not silent:
        print("")
        print("-----------------------------------------------------------------------")
        print("-     DOUBLY ROBUST AVERAGE REGRESSION ESTIMATION RESULTS             -")
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
