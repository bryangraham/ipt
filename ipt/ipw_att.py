# Load library dependencies
import numpy as np
import numpy.linalg

import scipy as sp
import scipy.optimize
import scipy.stats

import pandas as pd

# Import logit() command in ipt module since ipw_att() calls it
from .logit import logit
from .ols import ols

# Define ipw_att() function
#-----------------------------------------------------------------------------#
def ipw_att(D, Y, r_W, s_wgt=None, nocons=False, c_id=None, silent=False):
    
    """
    
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu
    DATE: Python 3.6 code on 15 July 2018        
    
    This function estimates the average treatment effect on the treated (ATT)
    by the method of inverse probability weighting as describe in, for example,
    Hirano, Imbens and Ridder (2003, Econometrica) and Imbens (2004, RESTAT). 
    A logit "series" propensity score model is assumed (i.e., p(W) = g(r(W)'delta)) 
    with g() a logit CDF and r(W) a vector of basis functions in always observed 
    pre-treatment covariates). We use the weighted least squares regression-based estimator
    described in, for example, Busso, DiNardo and McCrary (2014, RESTAT). This
    estimate automatically normalizes the implicit distribution function estimates
    associated with the IPW estimator to sum to one.
    

    INPUTS
    ------
    D         : N x 1 pandas.Series with ith element equal to 1 if ith unit in the merged
                sample is from the study population and zero if from the auxiliary
                population (i.e., D is the "treatment" indicator)
    Y         : N x 1  pandas.Series of observed outcomes: Y=DY1 + (1-D)Y0                  
    r_W       : r(W), N x 1+L pandas.DataFrame of functions of always observed covariates
                used as basis functions in the logit propensity score
    s_wgt     : N x 1 pandas.Series of sampling weights variable (optional)
    nocons    : If True, then do NOT add constant to r_W matrix
                (only set to True if user passes in r_W dataframe with a constant included)
    c_id      : N X 1 pandas.Series of unique `cluster' id values (assumed to be integer valued) (optional)
                NOTE: Default is to assume independent observations and report heteroscedastic robust 
                      standard errors
                NOTE: Data are assumed to be pre-sorted by groups.
    silent    : if silent = True display less optimization information and use
                lower tolerance levels (optional)

    OUTPUTS
    -------
    gamma_ipw         : IPW estimate of gamma (the ATT)
    vcov_gamma_ipw    : Estimated large sample variance of gamma
    pscore_test       : Newey-Tauchen-White balancing test statistic [statistic, dof, p-val]
    tilts             : numpy array with pi_eff, pi_s & pi_a as columns, sorted according
                        to the input data, and where                                     
                        pi_eff : Semiparametrically efficient estimate of F_s(W) 
                        pi_s   : Study sample tilt (treated)
                        pi_a   : Auxiliary sample tilt (control)
                        (implicit IPW distribution function estimates; note pi_eff = pi_s
                         under logit propensity score)
    exitflag          : 1 = success, 2 = can't compute MLE of p-score

    FUNCTIONS CALLED  : logit()                             (...logit_logl(), logit_score(), logit_hess()...)
    ----------------    ols()
    """
      
    # ----------------------------------------------------------------------------------- #
    # - STEP 1 : ORGANIZE DATA                                                          - #
    # ----------------------------------------------------------------------------------- #

    # Extract variable names from pandas data objects
    dep_var     = Y.name                  # Get dependent variable names
    r_W_names   = list(r_W.columns)       # Get r_W variable names
    
    # Create pointers to pandas objects transformed into appropriately sized numpy arrays
    Ds         = D.values.reshape((-1,1))       # Turn pandas.Series into N x 1 numpy array
    Ys         = Y.values.reshape((-1,1))       # Turn pandas.Series into N x 1 numpy array
    r_Ws       = r_W.values                     # Turn pandas.DataFrame into N x 1 + L numpy array
    
    # Extract basic information and set-up AST problem
    N         = len(D)                  # Number of units in sample  
    Ns        = np.sum(D)               # Number of study units in the sample (treated units) 
    Na        = N-Ns                    # Number of auxiliary units in the sample (control units)
    L         = np.shape(r_Ws)[1]    
   
    if nocons:
        L = L - 1                       # Dimension of r_W (excluding constant)
  
    # Add a constant to the regressor matrix (if needed)
    if not nocons:
        r_Ws      = np.concatenate((np.ones((N,1)), r_Ws), axis=1) 
        r_W_names = ['constant'] + r_W_names

    # Normalize weights to have mean one (if needed)
    if s_wgt is None:
        sw = 1 
    else:
        s_wgt_var = s_wgt.name                               # Get sample weight variable name
        sw = np.asarray(s_wgt/s_wgt.mean()).reshape((-1,1))  # Normalized sampling weights with mean one
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 2 : ESTIMATE PROPENSITY SCORE PARAMETER BY LOGIT ML                        - #
    # ----------------------------------------------------------------------------------- #
  
    try:
        if not silent:
            print("")
            print("--------------------------------------------------------------")
            print("- Computing propensity score by MLE                          -")
            print("--------------------------------------------------------------")
        
        # CMLE of p-score coefficients
        [delta_ml, vcov_delta_ml, hess_logl, score_i, p_W, success] = \
                                            logit(D, r_W, s_wgt=s_wgt, nocons=nocons, \
                                                  c_id=c_id, silent=silent, full=False)
        
        delta_ml                 = np.reshape(delta_ml,(-1,1))                 # Put delta_ml into 2-dimensional numpy form
        NQ                       = np.sum(sw * p_W)                            # Sum of fitted p-scores
        pi_eff                   = (sw * p_W) / NQ                             # Efficient estimate of F(W)
        
        # Implicit study and auxiliary sample tilts
        pi_s  = Ds * pi_eff / p_W            # Study (treated) ipw tilt 
                                             # (numerically identical to efficient tilt in logit case)
        pi_a  = (1-Ds) * (p_W / (1-p_W))     # Auxiliary (control) ipw tilt (normalized version)
        pi_a  = pi_a / pi_a.sum() 

        tilts        = np.concatenate((pi_eff, pi_s, pi_a), axis=1)            # Collect three sample tilts                        
        
    except:
        print("FATAL ERROR: exitflag = 2, unable to compute propensity score by maximum likelihood.")
        
        # Set all returnables to "None" and then exit function
        gamma_ipw      = None
        vcov_gamma_ipw = None
        pscore_test    = None
        tilts          = None
        exitflag       = 2
        
        return [gamma_ipw, vcov_gamma_ipw, pscore_test, tilts, exitflag]
            
    # ----------------------------------------------------------------------------------- #
    # - STEP 3 : SOLVE FOR IPW ESTIMATE OF GAMMA (i.e., ATT)                            - #
    # ----------------------------------------------------------------------------------- #
    
    omega_i = (sw * (Ds + (1-Ds) * (p_W / (1 - p_W))))   # ipw regression weights (N x 1 numpy 2d array)
    X = np.concatenate((np.ones((N,1)), Ds), axis=1)     # design matrix          (N x 2 numpy 2d array)

    # Compute ATT using weighted least squares approach 
    # This approach leads to automatic weight normalization (Imbens, 2004, RESTAT). 
    # See also Busso, Dinardo & McCrary (2014, RESTAT)
    [beta_ipw, vcov_beta_ipw] = ols(Y, pd.DataFrame(data=X, columns = ["constant", "D"]) , c_id=c_id, nocons=True, \
                                    s_wgt=pd.Series(omega_i.flatten()), silent=True)[0:2] 
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 4 : GENERALIZED INFORMATION MATRIX PROPENSITY SCORE SPECIFICATION TEST     - #
    # ----------------------------------------------------------------------------------- #
    
    # Compute Newey-Tauchen-White test statistic for correct specification of the propensity score
    # See Wooldridge (2010, Chapter 13.7) for a textbook description of the test
    g_i = sw * p_W * (Ds / p_W - (1-Ds)/(1-p_W)) * r_Ws              # N x (1 + L) matrix of balancing moments
    PI_hat = np.linalg.solve(score_i.T @ score_i, score_i.T @ g_i)   # K X L coefficient matrix
    g_resid_i = (g_i - score_i @ PI_hat).T                           # (1 + L) x N matrix of balancing moments residuals
    G   = np.sum(g_i, axis=0).reshape(-1,1)                          # Outer component of NTW statistic     
                                                                     # (i.e., bread of the sandwich)
    # NOTE: covariance matrix of NTW_resid_i is calculated below (i.e., middle component of test statistic)
       
    # ----------------------------------------------------------------------------------- #
    # - STEP 4 : FORM LARGE SAMPLE VARIANCE-COVARIANCE ESTIMATES                        - #
    # ----------------------------------------------------------------------------------- #

    # Form moment vector corresponding to full two step procedure
    m1 = (sw * (Ds - p_W) * r_Ws).T                     # 1+L       x N matrix of m_1 moments (logit scores)
    m2 = (omega_i * (Ys - X @ beta_ipw) * X).T          # 2         x N matrix of m_2 moments
    m  = np.concatenate((m1, m2), axis=0)               # 1 + L + 2 x N matrix of stacked moments                                                                 

    # Calculate covariance matrix of moment vector. Take into account any 
    # within-group dependence/clustering as needed
    
    if c_id is None:
        
        # Case 1: No cluster dependence to account for when constructing covariance matrix
        C   = N                                         # Number of clusters equals number of observations        
        fsc = N/(N - (1+L+2))                           # Finite-sample correction factor        
        V_m = fsc*(m @ m.T)/N                           # Moment covariance matrix
        V_NTW = (g_resid_i @ g_resid_i.T)               # middle component of the NTW test statistic 
        
    else:
        
        # Case 2: Need to correct for cluster dependence when constructing covariance matrix
    
        # Get number and unique list of clusters
        c_list  = np.unique(c_id)            
        C       = len(c_list)    

        # Calculate cluster-robust variance-covariance matrix of m
        # Sum moment vector within clusters
        sum_m   = np.empty((C,1+L+2))                  # initiate vector of cluster-summed moments
        sum_g   = np.empty((C,1+L))                    # initiate vector of cluster-summed NTW test moments
        
        for c in range(0,C):
           
            # sum of moments for units in c-th cluster
            b_cluster    = np.nonzero((c_id == c_list[c]))[0]                     # Observations in c-th cluster 
            sum_m[c,:]   = np.sum(m[np.ix_(range(0,1+L+2), b_cluster)], axis = 1) # Sum over "columns" within c-th cluster
            sum_g[c,:]   = np.sum(g_resid_i[np.ix_(range(0,1+L), b_cluster)], axis = 1)
            
        # Compute variance-covariance matrix of moment vector
        fsc = (N/(N - (1+L+2)))*(C/(C-1))             # Finite-sample correction factor
        V_m = fsc*(sum_m.T @ sum_m)/C                 # Variance-covariance of the summed moments
        V_NTW = (sum_g.T @ sum_g)                     # NTW middle test statistic component
    
    # Complete computation of Newey-Tauchen-White balancing test
    NTW          = G.T @ np.linalg.inv(V_NTW) @ G
    dof_NTW      = 1+L
    pval_NTW     = 1 - sp.stats.chi2.cdf(NTW[0,0], dof_NTW)
    pscore_test  = [NTW[0,0], dof_NTW, pval_NTW]
        
    # Form Jacobian matrix for entire parameter: theta = (delta, beta)
    e_V  = np.exp(r_Ws @ delta_ml)
  
    M1_delta = ((sw * (- e_V / (1 + e_V)**2) * r_Ws).T @ r_Ws)/N                             # 1 + L x 1 + L
    M2_delta = ((sw * (1 - Ds) * e_V * (Ys - X @ beta_ipw) * X).T @ r_Ws)/N                  # 2     x 1 + L 

    M2_beta = -((omega_i * X).T @ X)/N                                                       # 2     x 2
    
    M1 = np.hstack((M1_delta, np.zeros((1+L,2)))) 
    M2 = np.hstack((M2_delta, M2_beta))              
  
    # Concatenate Jacobian and compute inverse
    M_hat = (N/C)*np.vstack((M1, M2))              
    iM_hat = np.linalg.inv(M_hat)
   
    # Compute sandwich variance estimates
    vcov_theta_ipw  = (iM_hat @ V_m @ iM_hat.T)/C
    
    # Extract ATT point estimate and variance estimate
    gamma_ipw = beta_ipw[-1,0]
    vcov_gamma_ipw  = vcov_theta_ipw[-1,-1]       
    
    exitflag = 1 # IPW estimate of the ATT successfully computed!
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 6 : DISPLAY RESULTS                                                        - #
    # ----------------------------------------------------------------------------------- #
    
    if not silent:
        print("")
        print("-------------------------------------------------------------------------------------------")
        print("- Inverse Probability Weighting (IPW) estimates of the ATT                                -")
        print("-------------------------------------------------------------------------------------------")
        print("ATT:  " + "%10.6f" % gamma_ipw)    
        print("     (" + "%10.6f" % np.sqrt(vcov_gamma_ipw) + ")")
        print("")
        print("-------------------------------------------------------------------------------------------")
        if c_id is None:
            print("NOTE: Outcome variable   = " + dep_var)
            print("      Heteroscedastic-robust standard errors reported")
            print("      N1 = " "%0.0f" % Ns + ", N0 = " + "%0.0f" % Na)
        else:
            print("NOTE: Outcome variable   = " + dep_var)
            print("      Cluster-robust standard errors reported")
            print("      Cluster-variable   = " + c_id.name)
            print("      Number of clusters = " + "%0.0f" % C)        
            print("      N1 = " "%0.0f" % Ns + ", N0 = " + "%0.0f" % Na)
        if s_wgt is not None:
            print("NOTE: (Sampling) Weighted IPW estimates computed.")
            print("      Weight-variable   = " + s_wgt_var)    

        print("")
        print("-------------------------------------------------------------------------------------------")
        print("- Maximum likelihood estimates of the p-score                                             -")
        print("-------------------------------------------------------------------------------------------")
        print("")
        print("Independent variable       Coef.    ( Std. Err.) ")
        print("-------------------------------------------------------------------------------------------")
        
        c = 0
        for names in r_W_names:
            print(names.ljust(25) + "%10.6f" % delta_ml[c] + \
                             " (" + "%10.6f" % np.sqrt(vcov_theta_ipw[c,c]) + ")")
            c += 1
        
        # ----------------------------------------------------------- #
        # Calculate Kish's effect treatment and control sample size - #
        # ----------------------------------------------------------- #
        
        print("")
        print("-------------------------------------------------------------------------------------------")
        print("- Effective treatment and control sample sizes (Kish's approximation)                     -")
        print("-------------------------------------------------------------------------------------------")
        
        print("")
        j          = np.where(D)[0]        # find indices of treated units
        N_s_eff    = 1/np.sum(pi_s[j]**2)  # Kish's formula for effective sample size
        print("Kish's effective study/treated sample size = " "%0.0f" % N_s_eff)
        print("")
                
        print("Percentiles of N_s * pi_s distribution")
        quantiles  = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        qnt_pi_s   = np.percentile(Ns*pi_s[j],quantiles)
            
        c = 0
        for q in quantiles:
            print("%2.0f" % quantiles[c] + " percentile = " "%2.4f" % qnt_pi_s[c])
            c += 1     
        
        print("")
        print("-------------------------------------------------------------------------------------------")
        print("- NOTE: Any differences between effective and nominal treatment group                     -")   
        print("-       size is due to sampling weights alone.                                            -")
        print("-------------------------------------------------------------------------------------------")
        
        
        print("")
        j          = np.where(1-D)[0]      # find indices of control units
        N_a_eff    = 1/np.sum(pi_a[j]**2)  # Kish's formula for effective sample size
        print("Kish's effective auxiliary/control sample size = " "%0.0f" % N_a_eff)
        print("")
            
        print("Percentiles of N_a * pi_a distribution")
        quantiles  = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        qnt_pi_a   = np.percentile(Na*pi_a[j],quantiles)
               
        c = 0
        for q in quantiles:
            print("%2.0f" % quantiles[c] + " percentile = " "%2.4f" % qnt_pi_a[c])
            c += 1  
        
        print("")
        print("-------------------------------------------------------------------------------------------")
        print(" NOTE: Any differences between effective and nominal control group                        -")
        print("       size is due to sampling weights as well as covariate imbalance.                    -")
        print("-------------------------------------------------------------------------------------------")
        
        
        # ------------------------------------------- #
        # Construct "balancing" tables              - #
        # ------------------------------------------- #
                
        Na_wgt = np.sum(sw * (1-Ds) , axis = 0)
        Ns_wgt = np.sum(sw * Ds , axis = 0)              
            
        # Compute means of r_W across various distribution function estimates
        # Mean of r(W) across controls
        mu_r_D0      = np.sum(sw * (1-Ds) * r_Ws, axis = 0)/Na_wgt
        mu_r_D0_std  = np.sqrt(np.sum(sw * (1-Ds) * (r_Ws - mu_r_D0)**2, axis = 0)/Na_wgt)
            
        # Mean of r(W) across treated
        mu_r_D1      = np.sum(sw * Ds * r_Ws, axis = 0)/Ns_wgt
        mu_r_D1_std  = np.sqrt(np.sum(sw * Ds * (r_Ws - mu_r_D1)**2, axis = 0)/Ns_wgt)
            
        # Normalized mean differences across treatment and controls 
        # (cf., Imbens, 2015, Journal of Human Resources)
        NormDif_r    = (mu_r_D1 - mu_r_D0)/np.sqrt((mu_r_D1_std**2 + mu_r_D0_std**2)/2)    
            
        # Mean of r(W) across controls (auxiliary) after re-weighting
        mu_r_a       = np.sum(pi_a * r_Ws, axis = 0)
        mu_r_a_std   = np.sqrt(np.sum(pi_a * (r_Ws - mu_r_a)**2, axis = 0))
        
        # Mean of r(W) across treated (study) after re-weighting
        # NOTE: Coincides with semiparametrically efficient estimate under logit p-score    
        mu_r_s       = np.sum(pi_s * r_Ws, axis = 0)
        mu_r_s_std   = np.sqrt(np.sum(pi_s * (r_Ws - mu_r_s)**2, axis = 0))
        
        # Normalized mean differences across treatment and controls after re-weighting
        # (cf., Imbens, 2015, Journal of Human Resources)
        NormDif_r_rw = (mu_r_s - mu_r_a)/np.sqrt((mu_r_s_std**2 + mu_r_a_std**2)/2)    
        
            
        # Pre-balance table
        print("")
        print("Means & standard deviations of r_W                                                         ")
        print("-------------------------------------------------------------------------------------------")
        print("                            Treated (D = 1)        Control (D = 0)        Norm. Diff.      ")
        print("-------------------------------------------------------------------------------------------")
        c = 0
        for names in r_W_names:
            print(names.ljust(25) + "%8.4f" % mu_r_D1[c]  + " (" + "%8.4f" % mu_r_D1_std[c] + ")    " \
                                  + "%8.4f" % mu_r_D0[c]  + " (" + "%8.4f" % mu_r_D0_std[c] + ")    " \
                                  + "%8.4f" % NormDif_r[c]) 
            c += 1
            
        # Post-balance table
        print("")
        print("Means and standard deviations of r_W (post-re-weighting)                                   ")
        print("-------------------------------------------------------------------------------------------")
        print("                            Treated (D = 1)        Control (D = 0)        Norm. Diff.      ")
        print("-------------------------------------------------------------------------------------------")
        c = 0
        for names in r_W_names:
            print(names.ljust(25) + "%8.4f" % mu_r_s[c]    + " (" + "%8.4f" % mu_r_s_std[c]   + ")    " \
                                  + "%8.4f" % mu_r_a[c]    + " (" + "%8.4f" % mu_r_a_std[c]   + ")    " \
                                  + "%8.4f" % NormDif_r_rw[c]) 
            c += 1
            
        print("")
        print("Specification test for p-score (H_0 : E[p(W)*(D/p(W) - (1-D)/(1-p(W)))*r(W)] = 0)")
        print("-------------------------------------------------------------------------------------------")
        print("chi-square("+str(dof_NTW)+") = " + "%10.6f" % NTW + "   p-value: " + "% .6f" % pval_NTW)
        print("")
        print("-------------------------------------------------------------------------------------------")
        print("NOTE: Newey-Tauchen-White generalized information matrix balancing test.")
   

    return [gamma_ipw, vcov_gamma_ipw, pscore_test, tilts, exitflag]