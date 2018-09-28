# Load library dependencies
import numpy as np
import numpy.linalg

import scipy as sp
import scipy.optimize

# Import print_coef() command in ipt module since poisson() calls it
from .print_coef import print_coef

# Define poisson() function
#-----------------------------------------------------------------------------#
def poisson(Y, X, c_id=None, s_wgt=None, nocons=False, silent=False, full=True, phi_sv=None):
    
    """
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu
    DATE: 9 July 2018    
    
    OVERVIEW
    --------
    This function computes (quasi-) maximum likelihood estimates
    of the poission regression model with E[Y|X] = exp(X'phi)
    and Var(Y|X) = exp(X'phi). Huber (cluster) robust standard
    errors are reported. The variance restriction need not hold
    for consistency of phi (as a model of the conditional mean).
    
    INPUTS:
    -------
    Y        : n X 1 pandas.Series of dependent variable
    X        : n X K pandas.DataFrame of regressors (should include constant if desired)
    c_id     : n X 1 pandas.Series of unique `cluster' id values (assumed to be integer valued) (optional)
               NOTE: Default is to assume independent observations and report heteroscedastic robust 
                     standard errors
    s_wgt    : n X 1 pandas.Series of sampling weights variable (optional)
    nocons   : If True, then do NOT add constant to X matrix
                (only set to True if user passes in a dataframe with a constant included)
    silent   : if set equal to True, then suppress all outcome (optional)
    full     : if True report/print coefficient estimates
    phi_sv   : Vector of starting values for estimation
        
    OUTPUTS:
    --------
    phi_ml             : K x 1 vector of ML estimates of phi = (kappa, gamma')'
    vcov_hat           : K x K (cluster) Huber-robust variance-covariance estimate
    hess_logl          : Hessian matrix associated with log-likelihood
    score_i            : n x K matrix of likelihood score contributions for each unit
    ehat               : n x 1 matrix of E[Y|X] fitted values
    phi_res_ml.success : Flag for whether optimization successfully converged
    
    FUNCTIONS CALLED  : ...poisson_logl(), poisson_score(), poisson_hess()...
                        ...print_coef()...
    ----------------    
    """
    
    def poisson_logl(phi, Y, X, s_wgts):
                                
        # Form conditional mean function of Y given X for i=1,...,n
        phi  = np.reshape(phi,(-1,1))
        Xphi = X @ phi
        mu   = np.exp(Xphi)
       
        # Form negative weighted log-likelihood
        logl = -np.sum(s_wgts*(np.multiply(Y,Xphi) - mu), axis = 0)
              
        return logl
    
    def poisson_score(phi, Y, X, s_wgts):
        
        # Form conditional mean function of Y given X for i=1,...,n
        phi  = np.reshape(phi,(-1,1))
        Xphi = X @ phi
        mu   = np.exp(Xphi)
        
        # Form n x K vector of scores contributions
        score_i = -np.multiply(X,(Y - mu)*s_wgts)
              
        # Form K x 1 score vector (column sum of score_i)
        score = np.sum(score_i, axis = 0)
        score = np.ravel(score)
      
        return score
     
    def poisson_hess(phi, Y, X, s_wgts):
        
        # Form conditional mean function of Y given X for i=1,...,n
        phi  = np.reshape(phi,(-1,1))
        Xphi = X @ phi
        mu   = np.exp(Xphi)
               
        # Form K x K hessian matrix
        hess = np.multiply(X,mu).T @ (np.multiply(X,s_wgts))
        
        return hess
    
    def poisson_callback(delta):
        print("Value of -logL = "    + "%.6f" % poisson_logl(delta, Y, X, sw) + \
              ",  2-norm of score = "+ "%.6f" % numpy.linalg.norm(poisson_score(delta, Y, X, sw)))
        
    #--------------------------------------------------------#
    #- STEP 1: Set up estimation problem                    -#
    #--------------------------------------------------------#
    
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
    
    #------------------------------------------------------------------------#
    #- STEP 2: Compute MLE estimates of phi using BFGS quasi-Newton method  -#
    #------------------------------------------------------------------------#
    
    if not phi_sv:
        phi_sv    = np.zeros((K,))
    
    if silent==True:
        # Suppress optimization output, use coarser tolerance values
        phi_res_ml = sp.optimize.minimize(poisson_logl, phi_sv, args=(Y,X,sw), method='Newton-CG', \
                                          jac=poisson_score, hess=poisson_hess, \
                                          options={'xtol': 1e-8, 'maxiter': 1000, 'disp': False})
    else:
        # Show optimization output and use finer tolerance values
        # Derivative check at starting values
        grad_norm = sp.optimize.check_grad(poisson_logl, poisson_score, phi_sv,Y,X,sw)
        print('Fisher-Scoring Derivative check (2-norm): ' + "%.8f" % grad_norm)  
        phi_res_ml = sp.optimize.minimize(poisson_logl, phi_sv, args=(Y,X,sw), method='Newton-CG', \
                                          jac=poisson_score, hess=poisson_hess, callback = poisson_callback, \
                                          options={'xtol': 1e-10, 'maxiter': 10000, 'disp': True})
        
    #--------------------------------------------------------#
    #- STEP 3: Compute sandwich covariance matrix estimate  -#
    #--------------------------------------------------------#
    
    # Reshape coefficient array into K x 1 matrix
    phi_ml = np.reshape(phi_res_ml.x, (-1, 1))
    
    # N x 1 vector of mu = exp(X'phi_ML) values
    Xphi = X @ phi_ml
    ehat = np.exp(Xphi)
   
    # Compute n x K matrix of scores
    score_i = np.multiply(X*sw,Y - ehat)                    # n x K matrix of moment vectors
   
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
    
    # Compute variance-covariance matrix of phi_ML
    hess_logl = -poisson_hess(phi_ml, Y, X, sw)    # Negative of returned hessian is the desired object since
                                                    # the negative of the logL is minimized here
    iH     = np.linalg.inv(hess_logl)
    vcov_hat = iH @ omega @ iH.T                  
        
    if full:
        print("")
        print("-----------------------------------------------------------------------")
        print("-                     POISSON ESTIMATION RESULTS                      -")
        print("-----------------------------------------------------------------------")
        print("Dependent variable:        " + dep_var)
        print("Number of observations, n: " + "%0.0f" % n)
        print("")
        
        # Print coefficient estimates, standard errors and 95 percent confidence intervals
        print_coef(phi_ml, vcov_hat, var_names=ind_var, alpha=0.05)
        
        if c_id is None:
            print("NOTE: Huber-robust standard errors reported.")
        else:
            print("NOTE: Cluster-Huber-robust standard errors reported.")
            print("      Cluster-variable   = " + c_id.name)
            print("      Number of clusters = " + "%0.0f" % N)
            
        if s_wgt is not None:
            print("NOTE: (Sampling) Weighted MLE estimates computed.")
            print("      Weight-variable    = " + s_wgt.name)   
            
    return [phi_ml, vcov_hat, hess_logl, score_i, ehat, phi_res_ml.success]
