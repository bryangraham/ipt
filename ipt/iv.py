# Load library dependencies
import numpy as np
import numpy.linalg

import pandas as pd

# Import print_coef() command in ipt module since iv() calls it
from .print_coef import print_coef

# Define iv() function
#-----------------------------------------------------------------------------#
def iv(Y, X, Z, c_id=None, s_wgt=None, nocons=False, silent=False):
    
    """
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu  
    DATE: 26 May 2016, updated for Python 3.6 on 23 July 2018  
    
    This function returns linear instrumental variable coefficient estimates 
    associated with the linear fit of Y on X using Z as an instrument. 
    It reports either heteroscedastic- or cluster-robust standard errors as 
    directed by the user. The program also allows for the incorporation of 
    sampling weights. 
    
    
    INPUTS:
    -------
    Y        : n X 1 pandas.Series of dependent variable
    X        : n X K pandas.DataFrame of regressors (should include constant if desired)
    Z        : n X J pandas.DataFrame of instruments (should include constant if desired) J >= K
    c_id     : n X 1 pandas.Series of unique `cluster' id values (assumed to be integer valued) (optional)
               NOTE: Default is to assume independent observations and report heteroscedastic robust 
                     standard errors
    s_wgt    : n X 1 pandas.Series of sampling weights variable (optional)
    nocons   : If True, then do NOT add constant to X and Z matrices
               (only set to True if user passes in X and Z dataframes with constants included)
    silent   : if set equal to True, then suppress all outcome (optional)
    
    OUTPUTS:
    --------
    beta_hat      : K x 1 vector of instrumental variable estimates of beta
    vcov_beta_hat : K x K cluster-robust variance-covariance estimate
    exit_flag     : 1 => success, 2 => K < J (model under-identified)

    FUNCTIONS CALLED : None
    ----------------
    
    """
    
    n       = len(Y)                     # Number of observations
    K       = X.shape[1]                 # Number of regressors
    J       = Z.shape[1]                 # Number of instruments
    
    # Normalize weights to have mean one (if needed)
    if s_wgt is None:
        sw = 1 
    else:
        s_wgt_var = s_wgt.name                               # Get sample weight variable name
        sw = np.asarray(s_wgt/s_wgt.mean()).reshape((-1,1))  # Normalized sampling weights with mean one
    
    # Extract variable names from pandas data objects
    dep_var  = Y.name                   # Get dependent variable names
    ind_var  = list(X.columns)          # Get independent variable names
    inst_var = list(Z.columns)          # Get instrumental variable names
    
    # Transform pandas objects into appropriately sized numpy arrays
    Y       = Y.values.reshape((n,1))  # Turn pandas.Series into n x 1 numpy array
    X       = X.values                 # Turn pandas.DataFrame into n x K numpy array
    Z       = Z.values                 # Turn pandas.DataFrame into n x J numpy array
    
    # Add a constant to the regressor and instrument matrices (if needed)
    if not nocons:
        X        = np.concatenate((np.ones((n,1)), X), axis=1) 
        Z        = np.concatenate((np.ones((n,1)), Z), axis=1) 
        
        ind_var  = ['constant'] + ind_var
        inst_var = ['constant'] + inst_var
        
        K       += 1
        J       += 1
    
    if K == J:
        
        #----------------------------------#
        #- Exactly identified case        -#
        #----------------------------------#
        
        # Set-up problem
        ZX  = (sw * Z).T @ X
        ZY  = (sw * Z).T @ Y
        
        H   = ZX                    # K x K Hessian matrix
        C   = np.identity(K)        # K x J linear combination of moments set equal to zero
                                    # NOTE: Sample mean of moments in K = J case
        
        # Compute beta_hat  
        beta_hat = np.linalg.solve(ZX , ZY)
        exitflag = 1
    
    elif J > K:
        
        #----------------------------------#
        #- Overidentified identified case -#
        #----------------------------------#
        
        # Set-up problem
        ZZ  = (sw * Z).T @ Z
        ZX  = (sw * Z).T @ X
        ZY  = (sw * Z).T @ Y
        
        iZZ = np.linalg.inv(ZZ)
        H   = ZX.T @ iZZ @ ZX       # K x K Hessian matrix
        C   = ZX.T @ iZZ            # K x J linear combination of moments set equal to zero
        
        # Compute beta_hat  
        beta_hat = np.linalg.solve(H , C @ ZY)
        exitflag = 1
        
    else:
        
        #----------------------------------#
        #- Underidentified case           -#
        #----------------------------------#
        
        print("K < J, Insufficient number of columns in instrument matrix, Z (under-identified).")
        beta_hat      = None
        vcov_beta_hat = None
        exitflag      = 2
    
        return [beta_hat, vcov_hat, exitflag]
    
    
    # Compute estimate of variance-covariance matrix of the sample moment vector
    psi    = sw * Z * (Y - X @ beta_hat)     # n x J matrix of moment vectors
    
    if c_id is None: 
        
        # Calculate heteroscedastic robust variance-covariance matrix of psi
        fsc   = n/(n-K)                      # Finite-sample correction factor
        omega = fsc*(psi.T @ psi)            # J X J variance-covariance of the moments
        
    else:
        
        # Get number and unique list of clusters
        c_list  = np.unique(c_id)            
        N       = len(c_list)    
        
        # Calculate cluster-robust variance-covariance matrix of psi
        # Sum moment vector within clusters
        sum_psi = np.empty((N,J))
    
        for c in range(0,N):
           
            b_cluster    = np.nonzero((c_id == c_list[c]))[0]                   # Observations in c-th cluster 
            sum_psi[c,:] = np.sum(psi[np.ix_(b_cluster, range(0,J))], axis = 0) # Sum over rows within c-th cluster
            
        # Compute variance-covariance matrix of moment vector
        fsc   = (n/(n-K))*(N/(N-1))                      # Finite-sample correction factor
        omega = fsc*(sum_psi.T @ sum_psi)                # J X J variance-covariance of the summed moments
    
    
    # Compute variance-covariance matrix of beta_hat ("11 term formula")
    iH      = np.linalg.inv(H)                           # Inverse Hessian
    vcov_beta_hat = iH @ C @ omega @ C.T @ iH.T          # "(G'WG)^-1 X G'W' x OMEGA x GW x (G'WG)^-1"
    
    if not silent:
        print("")
        print("-----------------------------------------------------------------------")
        print("-                    LINEAR IV/2SLS ESTIMATION RESULTS                -")
        print("-----------------------------------------------------------------------")
        print("Dependent variable:        " + dep_var)
        print("Number of observations, n: " + "%0.0f" % n)
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
        
        print("-----------------------------------------------------------------------")
        
        endogenous = set(ind_var)  - set(inst_var) 
        excluded   = set(inst_var) - set(ind_var)
        
        print("")
        print("Endogenous right-hand-side regressors: ")
        print("-----------------------------------------------------------------------")
        for regressor in endogenous:
            print(regressor.ljust(25))
        print("")
        
        print("Excluded instruments: ")
        print("-----------------------------------------------------------------------")
        for instrument in excluded:
            print(instrument.ljust(25))

    return [beta_hat, vcov_beta_hat, exitflag]
