# Load library dependencies
import numpy as np
import numpy.linalg

import scipy as sp
import scipy.optimize

# Import print_coef() command in ipt module since logit() calls it
from .print_coef import print_coef

# Define logit() function
#-----------------------------------------------------------------------------#
def logit(D, X, s_wgt=None, nocons=False, c_id=None, silent=False, full=True):

    """
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu
    DATE: 26 May 2016, updated for Python 3.6 on 12 June 2018       
    
    This function computes the ML estimate of the logit binary choice model:
    Pr(D=1|X=x)= exp(x'delta)/[1+exp(x'delta)]. The function is able to handle
    sampling weights. It also reports cluster-robust standard errors if
    requested to do so. A constant is pre-pended to the regressor matrix
    unless nocons is set equal to True.

    INPUTS
    ------
    D      : n x 1 panda series of binary outcomes
    X      : X is a N x K pandas dataframe of regressors (without a constant, unless nocons=True)
    s_wgt  : n x 1 panda series of known sampling weights (optional)
    nocons : If True, then do NOT add constant to the design matrix 
    c_id   : N X 1 pandas series of unique `cluster' id values (assumed to be integer valued) (optional)
             NOTE: Default is to assume independent observations and report quasi-MLE robust 
                   standard errors (i.e., Huber formula)
    silent : when silent = True optimization output is suppressed and
             optimization is by Fisher-scoring with lower tolerances.
             Otherwise optimization output is 
             displayed with tighter convergence criteria imposed. 
    full   : if True report/print coefficient estimates       

    OUTPUTS
    -------
    gamma_ml             : (Quasi-) ML estimates of logit coefficients 
    vcov_hat             : Estimated asymptotic variance-covariance matrix
    hess_logl            : Hessian matrix associated with log-likelihood
    score_i              : n x K matrix of likelihood score contributions for each unit
    ehat                 : n x 1 vector of Pr(D=1|X) fitted probabilities
    delta_res_ml.success : Flag for whether optimization successfully converged

    FUNCTIONS CALLED     : ...logit_logl(), logit_score(), logit_hess()...
    ----------------    
    
    Functions called : ...logit_logl, logit_score, logit_hess, logit_callback...
                       ...print_coef()...
    
    """
    
    def logit_logl(delta, D, X, s_wgt):
        
        """
        Constructs logit log-likelihood.
        """
        
        delta      = np.reshape(delta,(-1,1))    # NOTE: scipy.optimize treats the parameter as a 1 dimensional array
        X_delta    = X @ delta                   #       The code below is based on treating it as 2 dimensional vector
        exp_Xdelta = np.exp(X_delta)             #       hence the reshaping.
        D * X_delta
        logl       = -np.sum(s_wgt * (D * X_delta - np.log(1+exp_Xdelta)))
        
        return logl
                        
    def logit_score(delta, D, X, s_wgt):
        
        """
        Constructs dim(delta) x 1 score vector associated with logit log-likelihood.
        NOTE: scipy.optimize requires that the score vector be returned as a 1 dimensional numpy array, NOT
              a 2 dimensional vector, hence the reshape and ravel calls at the start and end of the function.
        """
        
        delta      = np.reshape(delta,(-1,1))   # Reshape one-dimensional parameter array into two dimensional vector
        X_delta    = X @ delta                  # Form score
        exp_Xdelta = np.exp(X_delta)
        score      = -X.T @ (s_wgt * (D - (exp_Xdelta / (1+exp_Xdelta))))
        score      = np.ravel(score)            # Return score as 1 dimensional numpy array, not a 2 dimensional vector
        
        return score    
    
    def logit_hess(delta, D, X, s_wgt):
        
        """
        Constructs dim(delta) x dim(delta) hessian matrix associated with logit log-likelihood.
        """
        
        delta      = np.reshape(delta,(-1,1))
        X_delta    = X @ delta
        exp_Xdelta = np.exp(X_delta)
        hess       = (s_wgt * (exp_Xdelta / (1+exp_Xdelta)**2) * X).T @ X 
        
        return hess 
    
    def logit_callback(delta):
        print("Value of -logL = "    + "%.6f" % logit_logl(delta, Y, W, sw) + \
              ",  2-norm of score = "+ "%.6f" % numpy.linalg.norm(logit_score(delta, Y, W, sw)))
    
    #--------------------------------------------------------------------#
    #- STEP 1 : Organize data for estimation                            -#
    #--------------------------------------------------------------------#
                    
    (n, K) = np.shape(X)                        # Number of observations and covariates
    W      = np.asarray(X)                      # Create numpy views for regressors and        
    Y      = np.asarray(D).reshape((-1,1))      # outcome
        
    X_names = list(X.columns)                   # Get regressor/covariate labels names
        
    # Add a constant to the regressor matrix (if needed)
    if not nocons:
        W       = np.concatenate((np.ones((n,1)), W), axis=1)   
        K       = K + 1
        X_names = ['Constant'] + X_names
    
    # Normalize weights to have mean one (if needed)
    if s_wgt is None:
        sw = 1 
    else:
        sw = np.asarray(s_wgt/s_wgt.mean()).reshape((-1,1))  # Normalized sampling weights with mean one
  
    #--------------------------------------------------------------------#
    #- STEP 2 : Compute CMLE                                            -#
    #--------------------------------------------------------------------#                   
    
    # For starting values set constant to calibrate marginal probability of outcome and
    # all slope coefficients to zero     
    delta_sv    = np.zeros((K,))
    if not nocons:
        p_hat       = np.mean(D);
        delta_sv[0] = np.log(p_hat/(1-p_hat))
            
    if silent:
        # Suppress optimization output, use Fisher-Scoring, coarser tolerance values and fewer iterations
        # Compute MLE via Fisher-Scoring
        delta_res_ml = sp.optimize.minimize(logit_logl, delta_sv, args=(Y, W, sw), method='Newton-CG', \
                                            jac=logit_score, hess=logit_hess, \
                                            options={'xtol': 1e-6, 'maxiter': 1000, 'disp': False})
        
        delta_ml = delta_res_ml.x
        hess_logl = -logit_hess(delta_ml, Y, W, sw) # Negative of returned hessian is the desired object since
                                                    # the negative of the logL is minimized here
        
    else:
        # Show optimization output, use Fisher-Scoring, finer tolerance values and more iterations
        # Derivative check at starting values
        grad_norm = sp.optimize.check_grad(logit_logl, logit_score, delta_sv, Y, W, sw, epsilon = 1e-8)
        print('Fisher-Scoring Derivative check (2-norm): ' + "%.8f" % grad_norm)  
        
        # Solve for MLE
        delta_res_ml = sp.optimize.minimize(logit_logl, delta_sv, args=(Y, W, sw), method='Newton-CG', \
                                            jac=logit_score, hess=logit_hess, callback = logit_callback, \
                                            options={'xtol': 1e-12, 'maxiter': 10000, 'disp': True}) 
        delta_ml = delta_res_ml.x
        hess_logl = -logit_hess(delta_ml, Y, W, sw) # Negative of returned hessian is the desired object since
                                                    # the negative of the logL is minimized here
   
    #-------------------------------------------------------#
    #- Compute variance-covariance matrix                  -#
    #-------------------------------------------------------#
    
    # Compute n x K matrix of scores
    W_delta    = W @ delta_ml.reshape((-1,1))           
    exp_Wdelta = np.exp(W_delta)
    ehat       = exp_Wdelta / (1+exp_Wdelta)
    score_i    = -W * (sw * (Y - ehat))
       
    if c_id is None: 
        
        # Compute variance-covariance matrix of scores (unclustered)
        fsc   = n/(n-K)                                     # Finite-sample correction factor
        omega = fsc*(score_i.T @ score_i)                   # K X K variance-covariance of the moments
        
    else:
        
        # Get number and unique list of clusters
        c_list  = c_id.unique()            
        N       = len(c_list)    
        
        # Calculate cluster-robust variance-covariance matrix of score_i
        # Sum score vector within clusters
        sum_score = np.empty((N,K))
    
        for c in range(0,N):
            b_cluster      = np.nonzero((c_id == c_list[c]))[0]                       # Indices of observations in c-th cluster 
            sum_score[c,:] = np.sum(score_i[np.ix_(b_cluster, range(0,K))], axis = 0) # Sum over rows within c-th cluster
            # NOTE: Above line uses "open mesh" numpy indexing, see documentation
            
        # Compute variance-covariance matrix of scores (clustered)
        fsc   = (n/(n-K))*(N/(N-1))                         # Finite-sample correction factor
        omega = fsc*(sum_score.T @ sum_score)               # K X K variance-covariance of the summed moments
    
    # Compute variance-covariance matrix of delta_ml
    iH     = np.linalg.inv(hess_logl)
    vcov_hat = iH @ omega @ iH.T     
    

    if full:
        print("")
        print("-----------------------------------------------------------------------")
        print("-                     LOGIT ESTIMATION RESULTS                        -")
        print("-----------------------------------------------------------------------")
        print("Dependent variable:        " + D.name)
        print("Number of observations, n: " + "%0.0f" % n)
        print("")
        
        # Print coefficient estimates, standard errors and 95 percent confidence intervals
        print_coef(delta_ml, vcov_hat, var_names=X_names, alpha=0.05)
        
        if c_id is None:
            print("NOTE: Huber-robust standard errors reported.")
        else:
            print("NOTE: Cluster-Huber-robust standard errors reported.")
            print("      Cluster-variable   = " + c_id.name)
            print("      Number of clusters = " + "%0.0f" % N)
            
        if s_wgt is not None:
            print("NOTE: (Sampling) Weighted MLE estimates computed.")
            print("      Weight-variable    = " + s_wgt.name)   
        
    return [delta_ml, vcov_hat, hess_logl, score_i, ehat, delta_res_ml.success]       