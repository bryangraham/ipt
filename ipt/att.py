# Load library dependencies
import numpy as np
import numpy.linalg

import scipy as sp
import scipy.optimize
import scipy.stats

import pandas as pd

# Import logit() command in ipt module since att() calls it
from .logit import logit

# Define att() function
#-----------------------------------------------------------------------------#

def att(D, Y, r_W, t_W, study_tilt=True, rlgrz=1, s_wgt=None, nocons=False, c_id=None, silent=False):
    
    """
    
    AUTHOR: Bryan S. Graham, UC - Berkeley, bgraham@econ.berkeley.edu
    DATE: Python 2.7 code on 26 May 2016, updated for Python 3.6 on 15 July 2018        
    
    This function estimates the average treatment effect on the treated (ATT)
    using the "auxiliary-to-study tilting" (AST) method described by 
    Graham, Pinto and Egel (2016, Journal of Business and Economic Statistics). 
    The notation below mirrors that in the paper where possible. The Supplemental 
    Web Appendix of the paper describes the estimation algorithm implemented here 
    in detail. A copy of the paper and all supplemental appendices can be found 
    online at http://bryangraham.github.io/econometrics/

    INPUTS
    ------
    D         : N x 1 pandas.Series with ith element equal to 1 if ith unit in the merged
                sample is from the study population and zero if from the auxiliary
                population (i.e., D is the "treatment" indicator)
    Y         : N x 1  pandas.Series of observed outcomes                  
    r_W       : r(W), N x 1+L pandas.DataFrame of functions of always observed covariates
                -- these are the propensity score basis functions
    t_W       : t(W), N x 1+M pandas.DataFrame of functions of always observed covariates
                -- these are the balancing functions     
    study_tilt: If True compute the study sample tilt. This should be set to False 
                if all the elements in t(W) are also contained in h(W). In that 
                case the study_tilt coincides with its empirical measure.This 
                measure is returned in the pi_s vector when  study_tilt = False.
    rlgrz     : Regularization parameter. Should positive and less than or equal 
                to one. Smaller values correspond to less regularizations, but 
                may cause underflow problems when overlap is poor. The default 
                value will be adequate for most applications.
    s_wgt     : N x 1 pandas.Series of sampling weights variable (optional)
    nocons    : If True, then do NOT add constant to h_W and t_W matrix
                (only set to True if user passes in dataframes with constants included)
    c_id      : N X 1 pandas.Series of unique `cluster' id values (assumed to be integer valued) (optional)
                NOTE: Default is to assume independent observations and report heteroscedastic robust 
                      standard errors
                NOTE: Data are assumed to be pre-sorted by groups.
    silent    : if silent = True display less optimization information and use
                lower tolerance levels (optional)

    OUTPUTS
    -------
    gamma_ast         : AST estimate of gamma (the ATT)
    vcov_gamma_ast    : estimated large sample variance of gamma
    pscore_tests      : list of [study_test, auxiliary_test] where      
                        study_test     : ChiSq test statistic of H0 : lambda_s = 0; list with 
                                         [statistic, dof, p-val]
                                         NOTE: returns [None, None, None] if study_tilt = False
                        auxiliary_test : ChiSq test statistic of H0 : lambda_a = 0; list with 
                                         [statistic, dof, p-val]
    tilts             : numpy array with pi_eff, pi_s & pi_a as columns, sorted according
                        to the input data, and where                                     
                        pi_eff : Semiparametrically efficient estimate of F_s(W) 
                        pi_s   : Study sample tilt
                        pi_a   : Auxiliary sample tilt 
    exitflag          : 1 = success, 2 = can't compute MLE of p-score, 3 = can't compute study/treated tilt,
                        4 = can't compute auxiliary/control tilt

    FUNCTIONS CALLED  : logit()                             (...logit_logl(), logit_score(), logit_hess()...)
    ----------------    ast_crit(), ast_foc(), ast_soc()    (...ast_phi()...)
    """
    
    def ast_phi(lmbda, t_W, p_W_index, NQ, rlgrz):
        
        """
        This function evaluates the regularized phi(v) function for 
        the logit propensity score case (as well as its first and 
        second derivatives) as described in the Supplemental
        Web Appendix of Graham, Pinto and Egel (2016, JBES).

        INPUTS
        ------
        lmbda         : vector of tilting parameters
        t_W           : vector of balancing moments
        p_W_index     : index of estimated logit propensity score
        NQ            : sample size times the marginal probability of missingness
        rlgrz         : Regularization parameter. See discussion in main header.
        
        OUTPUTS
        -------
        phi, phi1, phi2 : N x 1 vectors with elements phi(p_W_index + lmbda't_W)
                          and its first and second derivatives w.r.t to 
                          v = p_W_index + lmbda't_W
        """
        
        # Adjust the NQ cut-off value used for quadratic extrapolation according
        # to the user-defined rlgrz parameter
        NQ =  NQ*rlgrz        
        
        # Coefficients on quadratic extrapolation of phi(v) used to regularize 
        # the problem
        c = -(NQ - 1)
        b = NQ + (NQ - 1)*np.log(1/(NQ - 1))
        a = -(NQ - 1)*(1 + np.log(1/(NQ - 1)) + 0.5*(np.log(1/(NQ - 1)))**2) 
        
        v_star = np.log(1/(NQ - 1)) 

        # Evaluation of phi(v) and derivatives
        v          =  p_W_index + t_W @ lmbda
        phi        =  (v>v_star) * (v - np.exp(-v))   + (v<=v_star) * (a + b*v + 0.5*c*v**2)
        phi1       =  (v>v_star) * (1 + np.exp(-v))   + (v<=v_star) * (b + c*v)
        phi2       =  (v>v_star) * (  - np.exp(-v))   + (v<=v_star) * c
          
        return [phi, phi1, phi2]

    def ast_crit(lmbda, D, p_W, p_W_index, t_W, NQ, rlgrz, s_wgt):
        
        """
        This function constructs the AST criterion function
        as described in Graham, Pinto and Egel (2016, JBES).
        
        INPUTS
        ------
        lmbda         : vector of tilting parameters
        D             : N x 1 treatment indicator vector
        p_W           : N x 1 MLEs of the propensity score
        p_W_index     : index of estimated logit propensity score
        t_W           : vector of balancing moments
        NQ            : sample size times the marginal probability of missingness
        rlgrz         : Regularization parameter. See discussion in main header.
        s_wgt         : N x 1 vector of known sampling weights (optional)
        
        OUTPUTS
        -------
        crit          : AST criterion function at passed parameter values
        
        Functions called : ast_phi()
        """
        
        lmbda   = np.reshape(lmbda,(-1,1))                                # make lmda 2-dimensional object
        [phi, phi1, phi2] = ast_phi(lmbda, t_W, p_W_index, NQ, rlgrz)     # compute phi and 1st/2nd derivatives
        crit    = -np.sum(s_wgt * (D * phi - (t_W @ lmbda)) * (p_W / NQ)) # AST criterion (scalar)
        
        return crit
    
    def ast_foc(lmbda, D, p_W, p_W_index, t_W, NQ, rlgrz, s_wgt):
        
        """
        Returns first derivative vector of AST criterion function with respect
        to lmbda. See the header for ast_crit() for description of parameters.
        """
        
        lmbda   = np.reshape(lmbda,(-1,1))                              # make lmda 2-dimensional object
        [phi, phi1, phi2] = ast_phi(lmbda, t_W, p_W_index, NQ, rlgrz)   # compute phi and 1st/2nd derivatives
        foc     = -(t_W.T @ (s_wgt * (D * phi1 - 1) * (p_W / NQ)))      # AST gradient (1+M x 1 vector)
        foc     = np.ravel(foc)                                         # make foc 1-dimensional numpy array
        
        return foc
    
    def ast_soc(lmbda, D, p_W, p_W_index, t_W, NQ, rlgrz, s_wgt):
        
        """
        Returns hessian matrix of AST criterion function with respect
        to lmbda. See the header for ast_crit() for description of parameters.
        """
        
        lmbda   = np.reshape(lmbda,(-1,1))                            # make lmda 2-dimensional object
        [phi, phi1, phi2] = ast_phi(lmbda, t_W, p_W_index, NQ, rlgrz) # compute phi and 1st/2nd derivatives
        soc     = -(((s_wgt * D * phi2 * (p_W / NQ)) * t_W).T @ t_W)  # AST hessian (note use of numpy broadcasting rules)
                                                                      # (1 + M) x (1 + M) "matrix" (numpy array) 
        return [soc]
    
    def ast_study_callback(lmbda):
        print("Value of ast_crit = "   + "%.6f" % ast_crit(lmbda, Ds, p_W, p_W_index, t_Ws, NQ, rlgrz, sw) + \
              ",  2-norm of ast_foc = "+ "%.6f" % numpy.linalg.norm(ast_foc(lmbda, Ds, p_W, p_W_index, t_Ws, \
                                                                            NQ, rlgrz, sw)))
    
    def ast_auxiliary_callback(lmbda):
        print("Value of ast_crit = "   + "%.6f" % ast_crit(lmbda, 1-Ds, p_W, -p_W_index, t_Ws, NQ, rlgrz, sw) + \
              ",  2-norm of ast_foc = "+ "%.6f" % numpy.linalg.norm(ast_foc(lmbda, 1-Ds, p_W, -p_W_index, t_Ws, \
                                                                            NQ, rlgrz, sw)))
  
    # ----------------------------------------------------------------------------------- #
    # - STEP 1 : ORGANIZE DATA                                                          - #
    # ----------------------------------------------------------------------------------- #

    # Extract variable names from pandas data objects
    dep_var     = Y.name                  # Get dependent variable names
    r_W_names   = list(r_W.columns)       # Get r_W variable names
    t_W_names   = list(t_W.columns)       # Get t_W variable names
    
    # Create pointers to pandas objects transformed into appropriately sized numpy arrays
    Ds         = D.values.reshape((-1,1)) # Turn pandas.Series into N x 1 numpy array
    Ys         = Y.values.reshape((-1,1)) # Turn pandas.Series into N x 1 numpy array
    r_Ws       = r_W.values               # Turn pandas.DataFrame into N x 1 + L numpy array
    t_Ws       = t_W.values               # Turn pandas.DataFrame into N x 1 + M numpy array

    # Extract basic information and set-up AST problem
    N         = len(D)                    # Number of units in sample  
    Ns        = np.sum(D)                 # Number of study units in the sample (treated units) 
    Na        = N-Ns                      # Number of auxiliary units in the sample (control units)
    M         = np.shape(t_Ws)[1]   
    L         = np.shape(r_Ws)[1]    
   
    if nocons:
        M = M - 1                         # Dimension of t_W (excluding constant)
        L = L - 1                         # Dimension of r_W (excluding constant)
        
    DY        = Ds * Ys                   # D*Y, N x 1  vector of observed outcomes for treated/study units
    mDX       = (1-Ds) * Ys               # (1-D)*X, N x 1  vector of observed outcomes for non-treated/auxiliary units 

    # Add a constant to the regressor matrix (if needed)
    if not nocons:
        r_Ws      = np.concatenate((np.ones((N,1)), r_Ws), axis=1) 
        r_W_names = ['constant'] + r_W_names
        t_Ws      = np.concatenate((np.ones((N,1)), t_Ws), axis=1) 
        t_W_names = ['constant'] + t_W_names
    
    # Normalize weights to have mean one (if needed)
    if s_wgt is None:
        sw = 1 
    else:
        s_wgt_var = s_wgt.name                 # Get sample weight variable name
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
        [delta_ml, vcov_delta_ml, hess_logl, score_i, p_W, _] = \
                                            logit(D, r_W, s_wgt=s_wgt, nocons=nocons, \
                                                  c_id=c_id, silent=silent, full=False)
        
        delta_ml                 = np.reshape(delta_ml,(-1,1))                 # Put delta_ml into 2-dimensional form
        p_W_index                = r_Ws @ delta_ml                             # Fitted p-score index 
        NQ                       = np.sum(sw * p_W)                            # Sum of fitted p-scores
        pi_eff                   = (sw * p_W) / NQ                             # Efficient estimate of F(W)
    
    except:
        print("FATAL ERROR: exitflag = 2, unable to compute propensity score by maximum likelihood.")
        
        # Set all returnables to "None" and then exit function
        gamma_ast      = None
        vcov_gamma_ast = None
        pscore_tests   = None    
        tilts          = None
        exitflag       = 2
        
        return [gamma_ast, vcov_gamma_ast, pscore_tests, tilts, exitflag]
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 3 : SOLVE FOR AST TILTING PARAMETERS                                       - #
    # ----------------------------------------------------------------------------------- 

    # Set optimization parameters
    if silent:
        # Use Newton-CG solver with vector of zeros as starting values, 
        # low tolerance levels, and smaller number of allowed iterations.
        # Hide iteration output.
        options_set = {'xtol': 1e-8, 'maxiter': 1000, 'disp': False}
    else:
        # Use Newton-CG solver with vector of zeros as starting values, 
        # high tolerance levels, and larger number of allowed iterations.
        # Show iteration output.
        options_set = {'xtol': 1e-12, 'maxiter': 10000, 'disp': True}
        
    lambda_sv = np.zeros(1+M) # use vector of zeros as starting values
  
    #------------------------------#
    #- STUDY TILT                 -#
    #------------------------------#

    # NOTE: Only compute the study_tilt if directed to do so (this is the default). The study_tilt
    #       doesn't need to be computed if all the elements of t(W) are also included in h(W). It
    #       is the users responsibility to check this condition.
    
    if study_tilt:
        # -------------------------------------------------- #
        # - CASE 1: Non-trivial study sample tilt required - #
        # -------------------------------------------------- #
            
        # Compute lamba_s_hat (study or treated sample tilting parameters)
        try:
            if not silent:
                print("")
                print("--------------------------------------------------------------")
                print("- Computing study/treated sample tilt                        -")
                print("--------------------------------------------------------------")
            
                # Derivative check at starting values
                grad_norm = sp.optimize.check_grad(ast_crit, ast_foc, lambda_sv, Ds, p_W, \
                                                   p_W_index, t_Ws, NQ, rlgrz, \
                                                   sw, epsilon = 1e-12)
                print('Study sample tilt derivative check (2-norm): ' + "%.8f" % grad_norm)
                
                # Solve for tilting parameters
                lambda_s_res = sp.optimize.minimize(ast_crit, lambda_sv, args=(Ds, p_W, p_W_index, \
                                                                               t_Ws, NQ, rlgrz, sw), \
                                                    method='Newton-CG', jac=ast_foc, hess=ast_soc, \
                                                    callback = ast_study_callback, options=options_set)
            else:
                # Solve for tilting parameters
                lambda_s_res = sp.optimize.minimize(ast_crit, lambda_sv, args=(Ds, p_W, p_W_index, \
                                                                               t_Ws, NQ, rlgrz, sw), \
                                                    method='Newton-CG', jac=ast_foc, hess=ast_soc, \
                                                    options=options_set)
        except:
            print("FATAL ERROR: exitflag = 3, Unable to compute the study/treated vector of tilting parameters.")
        
            # Set all returnables to "None" and then exit function
            gamma_ast      = None
            vcov_gamma_ast = None
            pscore_tests   = None    
            tilts          = None
            exitflag       = 3
        
            return [gamma_ast, vcov_gamma_ast, pscore_tests, tilts, exitflag]
        
        # Collect study tilt estimation results needed below
        lambda_s_hat = np.reshape(lambda_s_res.x,(-1,1))                           # study/treated sample tilting 
                                                                                   # parameter estimates
        p_W_s = (1+np.exp(-(p_W_index) - (t_Ws @ lambda_s_hat)))**-1               # study/treated sample tilted p-score
        pi_s  = Ds * pi_eff / p_W_s                                                # study/treated sample tilt 
    
    else:
        # ------------------------------------------ #
        # - CASE 2: Study sample tilt NOT required - #
        # ------------------------------------------ #
        
        if not silent:
            print("")
            print("----------------------------------------------------------------------")
            print("- Tilt of study sample not requested by user (study_tilt = False).   -")
            print("- Validity of this requires all elements of t(W) to be elements of   -")
            print("- h(W) as well. User is advised to verify this condition.            -")
            print("----------------------------------------------------------------------")
            print("")
        
        # Collect study tilt objects needed below
        lambda_s_hat = np.reshape(lambda_sv ,(-1,1)) # study/treated sample tilting parameters set equal to zero
        p_W_s = p_W                                  # study/treated sample tilted p-score equals actual score
        pi_s  = Ds * pi_eff / p_W_s                  # set pi_s to "empirical measure" of study sub-sample 
                                                     # (w/o sampling weights this puts mass 1/Ns on each study unit)
   
    #------------------------------#
    #- AUXILIARY TILT             -#
    #------------------------------#
    
    # Compute lamba_a_hat (auxiliary or control sample tilting parameters)
    try:
        if not silent:
            print("")
            print("--------------------------------------------------------------")
            print("- Computing auxiliary/control sample tilt                    -")
            print("--------------------------------------------------------------")
            
            # Derivative check at starting values
            grad_norm = sp.optimize.check_grad(ast_crit, ast_foc, lambda_sv, 1-Ds, p_W, \
                                               -p_W_index, t_Ws, NQ, rlgrz, \
                                               sw, epsilon = 1e-12)
            print('Auxiliary sample tilt derivative check (2-norm): ' + "%.8f" % grad_norm) 
            
            # Solve for tilting parameters
            lambda_a_res = sp.optimize.minimize(ast_crit, lambda_sv, args=(1-Ds, p_W, -p_W_index, \
                                                                           t_Ws, NQ, rlgrz, sw), \
                                                method='Newton-CG', jac=ast_foc, hess=ast_soc, \
                                                callback = ast_auxiliary_callback, options=options_set)    
        else:     
            # Solve for tilting parameters
            lambda_a_res = sp.optimize.minimize(ast_crit, lambda_sv, args=(1-Ds, p_W, -p_W_index, \
                                                                           t_Ws, NQ, rlgrz, sw), \
                                                method='Newton-CG', jac=ast_foc, hess=ast_soc, \
                                                options=options_set)
    except:
        print("FATAL ERROR: exitflag = 4, Unable to compute the auxiliary/control vector of tilting parameters.")
        
        # Set returnables to "None" and then exit function
        gamma_ast      = None
        vcov_gamma_ast = None
        pscore_tests   = None    
        tilts          = None
        exitflag       = 4
        
        return [gamma_ast, vcov_gamma_ast, pscore_tests, tilts, exitflag]
    
    # Collect auxiliary tilt estimation results needed below
    lambda_a_hat = -np.reshape(lambda_a_res.x,(-1,1))                          # auxiliary/control sample tilting 
                                                                               # parameter estimates 
    p_W_a = (1+np.exp(-(p_W_index) - (t_Ws @ lambda_a_hat)))**-1               # auxiliary sample tilted p-score
    pi_a  = (1-Ds) * (pi_eff / (1-p_W_a))                                      # auxiliary sample tilt
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 4 : SOLVE FOR AST ESTIMATE OF GAMMA (i.e., ATT)                            - #
    # ----------------------------------------------------------------------------------- #

    # AST estimate of gamma -- the ATT %
    gamma_ast = np.sum(sw * p_W * ((Ds / p_W_s) * DY - (1-Ds) / (1-p_W_a) * mDX))/NQ;
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 5 : FORM LARGE SAMPLE VARIANCE-COVARIANCE ESTIMATES                        - #
    # ----------------------------------------------------------------------------------- #

    # Form moment vector corresponding to full three step procedure
    m1 = (sw * (Ds - p_W) * r_Ws).T                          # 1+L x N matrix of m_1 moments (logit scores)
    m2 = (sw * ((1 - Ds) / (1 - p_W_a) - 1) * p_W * t_Ws).T  # 1+M x N matrix of m_2 moments
    m3 = (sw * (Ds / p_W_s - 1) * p_W * t_Ws).T              # 1+M x N matrix of m_3 moments
    m4 = (sw * p_W * ((Ds / p_W_s) * DY - ((1-Ds) / (1-p_W_a)) * (mDX+gamma_ast))).T  # 1 x N matrix of m_4 moments
    m  = np.concatenate((m1, m2, m3, m4), axis=0)            # 1 + L + 2(1 + M) + 1 x N matrix of all moments                                                                 

    # Calculate covariance matrix of moment vector. Take into account any 
    # within-group dependence/clustering as needed
    if c_id is None:
        
        # Case 1: No cluster dependence to account for when constructing covariance matrix
        C   = N                                         # Number of clusters equals number of observations        
        fsc = N/(N - (1+L+2*(1+M)+1))                   # Finite-sample correction factor        
        V_m = fsc*(m @ m.T)/N
        
    else:
        
        # Case 2: Need to correct for cluster dependence when constructing covariance matrix
    
        # Get number and unique list of clusters
        c_list  = np.unique(c_id)            
        C       = len(c_list)    

        # Calculate cluster-robust variance-covariance matrix of m
        # Sum moment vector within clusters
        sum_m   = np.empty((C,1+L+2*(1+M)+1))           # initiate vector of cluster-summed moments
        
        for c in range(0,C):
           
            # sum of moments for units in c-th cluster
            b_cluster    = np.nonzero((c_id == c_list[c]))[0]                             # Observations in c-th cluster 
            sum_m[c,:]   = np.sum(m[np.ix_(range(0,1+L+2*(1+M)+1), b_cluster)], axis = 1) # Sum over "columns" within c-th cluster
            
        # Compute variance-covariance matrix of moment vector
        fsc = (N/(N - (1+L+2*(1+M)+1)))*(C/(C-1))     # Finite-sample correction factor
        V_m = fsc*(sum_m.T @ sum_m)/C                 # Variance-covariance of the summed moments        
        
        
    # Form Jacobian matrix for entire parameter: theta = (rho, delta, lambda, gamma)
    e_V  = np.exp(np.dot(r_Ws, delta_ml))
    e_Va = np.exp(np.dot(r_Ws, delta_ml) + np.dot(t_Ws, lambda_a_hat))
    e_Vs = np.exp(np.dot(r_Ws, delta_ml) + np.dot(t_Ws, lambda_s_hat))

    M1_delta = np.dot((sw * (- e_V / (1 + e_V)**2) * r_Ws).T, r_Ws)/N                                                    # 1 + L x 1 + L
    M2_delta = np.dot((sw * ((1 - Ds) / (1 - p_W_a) - 1) * (e_V / (1 + e_V)**2) * t_Ws).T, r_Ws)/N                       # 1 + M x 1 + L     
    M3_delta = np.dot((sw * (Ds / p_W_s - 1) * (e_V / (1 + e_V)**2) * t_Ws).T, r_Ws)/N                                   # 1 + M x 1 + L     
    M4_delta = np.dot((sw * (e_V / (1 + e_V)**2) * \
                      ((Ds / p_W_s) * DY - ((1 - Ds) / (1 - p_W_a)) * (mDX + gamma_ast))).T, r_Ws)/N                     # 1     x 1 + L    

    M2_lambda_a = np.dot(( sw * ((1 - Ds) / (1 - p_W_a)**2) * p_W * (e_Va / (1 + e_Va)**2) * t_Ws).T, t_Ws)/N            # 1 + M x 1 + M
    M4_lambda_a = np.dot((-sw * ((1 - Ds) / (1 - p_W_a)**2) * p_W * (mDX+gamma_ast) * (e_Va / (1 + e_Va)**2)).T, t_Ws)/N # 1     x 1 + M                                    

    M3_lambda_s = np.dot((-sw * (Ds / p_W_s**2) * p_W * (e_Vs / (1 + e_Vs)**2) * t_Ws).T, t_Ws)/N                        # 1 + M x 1 + M
    M4_lambda_s = np.dot((-sw * (Ds / p_W_s**2) * p_W * DY * (e_Vs / (1 + e_Vs)**2)).T, t_Ws)/N                          # 1     x 1 + M

    M4_gamma    = -(NQ/N).reshape(1,1)                                                                                   # 1     x 1  
    
    M1 = np.hstack((M1_delta, np.zeros((1+L,1+M)), np.zeros((1+L,1+M)), np.zeros((1+L,1)))) 
    M2 = np.hstack((M2_delta, M2_lambda_a,         np.zeros((1+M,1+M)), np.zeros((1+M,1))))  
    M3 = np.hstack((M3_delta, np.zeros((1+M,1+M)), M3_lambda_s,         np.zeros((1+M,1))))    
    M4 = np.hstack((M4_delta, M4_lambda_a,         M4_lambda_s,         M4_gamma))              
    
    # Concatenate Jacobian and compute inverse
    M_hat  = (N/C)*np.vstack((M1, M2, M3, M4))              
    iM_hat = np.linalg.inv(M_hat)
   
    # Compute sandwich variance estimates
    vcov_theta_ast  = (iM_hat @ V_m @ iM_hat.T)/C
    vcov_gamma_ast  = vcov_theta_ast[-1,-1]       
    
    exitflag = 1 # AST estimate of the ATT successfully computed!
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 6 : COMPUTE TEST STATISTICS BASED ON TILTING PARAMETER                     - #
    # ----------------------------------------------------------------------------------- #
    
    # Compute propensity score specification test based on study tilt (if applicable)
    if study_tilt:
        iV_lambda_s = np.linalg.inv(vcov_theta_ast[1+L:1+L+1+M,1+L:1+L+1+M])
        ps_test_st  = np.dot(np.dot(lambda_s_hat.T, iV_lambda_s), lambda_s_hat)
        dof_st      = len(lambda_s_hat)
        pval_st     = 1 - sp.stats.chi2.cdf(ps_test_st[0,0], dof_st)
        study_test  = [ps_test_st[0,0], dof_st, pval_st]
    else:
        study_test  = [None, None, None]
        
    # Compute propensity score specification test based on auxiliary tilt (always done)
    iV_lambda_a    = np.linalg.inv(vcov_theta_ast[1+L+1+M:1+L+1+M+1+M,1+L+1+M:1+L+1+M+1+M])
    ps_test_at     = np.dot(np.dot(lambda_a_hat.T, iV_lambda_a), lambda_a_hat)
    dof_at         = len(lambda_a_hat)
    pval_at        = 1 - sp.stats.chi2.cdf(ps_test_at[0,0], dof_at)   
    auxiliary_test = [ps_test_at[0,0], dof_at, pval_at]
    
    # ----------------------------------------------------------------------------------- #
    # - STEP 7 : DISPLAY RESULTS                                                        - #
    # ----------------------------------------------------------------------------------- #
    
    if not silent:
        print("")
        print("-------------------------------------------------------------------------------------------")
        print("- Auxiliary-to-Study (AST) estimates of the ATT                                           -")
        print("-------------------------------------------------------------------------------------------")
        print("ATT:  " + "%10.6f" % gamma_ast)    
        print("     (" + "%10.6f" % np.sqrt(vcov_gamma_ast) + ")")
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
            print("NOTE: (Sampling) Weighted AST estimates computed.")
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
                             " (" + "%10.6f" % np.sqrt(vcov_theta_ast[c,c]) + ")")
            c += 1
                
        print("")
        print("-------------------------------------------------------------------------------------------")
        print("- Tilting parameter estimates                                                             -")
        print("-------------------------------------------------------------------------------------------")
        
        if study_tilt:
            print("")
            print("TREATED (study) sample tilt")
            print("-------------------------------------------------------------------------------------------")
            print("")
            print("Independent variable       Coef.    ( Std. Err.) ")
            print("-------------------------------------------------------------------------------------------")
                
            c = 0
            for names in t_W_names:
                print(names.ljust(25) + "%10.6f" % lambda_s_hat[c] + \
                                 " (" + "%10.6f" % np.sqrt(vcov_theta_ast[1+L+c,1+L+c]) + ")")
                c += 1
                    
            print("")
            print("Specification test for p-score (H_0 : lambda_s = 0)")
            print("-------------------------------------------------------------------------------------------")
            print("chi-square("+str(dof_st)+") = " + "%10.6f" % ps_test_st + "   p-value: " + "% .6f" % pval_st) 
                
            print("")
            print("Summary statistics study/treated re-weighting")
            print("-------------------------------------------------------------------------------------------")
                
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
            
        else:
            print("")
            print("--------------------------------------------------------")
            print("- NOTE: Study tilt not computed (study_tilt = False).  -")
            print("-       Components of t(W) assumed to be also in h(W). -")
            print("--------------------------------------------------------")
            print("")
                
        print("")
        print("CONTROL (auxiliary) sample tilt")
        print("-------------------------------------------------------------------------------------------")
        print("")
        print("Independent variable       Coef.    ( Std. Err.) ")
        print("-------------------------------------------------------------------------------------------")
            
        c = 0
        for names in t_W_names:
            print(names.ljust(25) + "%10.6f" % lambda_a_hat[c] + \
                             " (" + "%10.6f" % np.sqrt(vcov_theta_ast[1+L+1+M+c,1+L+1+M+c]) + ")")
            c += 1 
                
        print("")
        print("Specification test for p-score (H_0 : lambda_a = 0)")
        print("-------------------------------------------------------------------------------------------")
        print("chi-square("+str(dof_at)+") = " + "%10.6f" % ps_test_at + "   p-value: " + "% .6f" % pval_at)
            
        print("")
        print("Summary statistics auxiliary/control re-weighting")
        print("-------------------------------------------------------------------------------------------")
                
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
              
        # ------------------------------------------- #
        # Construct "exact balancing" table         - #
        # ------------------------------------------- #
            
        Na_wgt = np.sum(sw * (1-Ds) , axis = 0)
        Ns_wgt = np.sum(sw * Ds , axis = 0)              
            
        # Compute means of t_W across various distribution function estimates
        # Mean of t(W) across controls
        mu_t_D0      = np.sum(sw * (1-Ds) * t_Ws, axis = 0)/Na_wgt
        mu_t_D0_std  = np.sqrt(np.sum(sw * (1-Ds) * (t_Ws - mu_t_D0)**2, axis = 0)/Na_wgt)
            
        # Mean of t(W) across treated
        mu_t_D1      = np.sum(sw * Ds * t_Ws, axis = 0)/Ns_wgt
        mu_t_D1_std  = np.sqrt(np.sum(sw * Ds * (t_Ws - mu_t_D1)**2, axis = 0)/Ns_wgt)
            
        # Normalized mean differences across treatment and controls 
        # (cf., Imbens, 2015, Journal of Human Resources)
        NormDif_t    = (mu_t_D1 - mu_t_D0)/np.sqrt((mu_t_D1_std**2 + mu_t_D0_std**2)/2)    
                                    
        # Semiparametrically efficient estimate of mean of t(W) across treated
        mu_t_eff     = np.sum(pi_eff * t_Ws, axis = 0)
        mu_t_eff_std = np.sqrt(np.sum(pi_eff * (t_Ws - mu_t_eff)**2, axis = 0))
            
        # Mean of t(W) across controls after re-weighting
        mu_t_a     = np.sum(pi_a * t_Ws, axis = 0)
        mu_t_a_std = np.sqrt(np.sum(pi_a * (t_Ws - mu_t_a)**2, axis = 0))
            
        # Mean of t(W) across treated after re-weighting
        mu_t_s     = np.sum(pi_s * t_Ws, axis = 0)
        mu_t_s_std = np.sqrt(np.sum(pi_s * (t_Ws - mu_t_s)**2, axis = 0))
            
        # Pre-balance table
        print("")
        print("Means & standard deviations of t_W (pre-balance)                                           ")
        print("-------------------------------------------------------------------------------------------")
        print("                            Treated (D = 1)        Control (D = 0)        Norm. Diff.      ")
        print("-------------------------------------------------------------------------------------------")
        c = 0
        for names in t_W_names:
            print(names.ljust(25) + "%8.4f" % mu_t_D1[c]  + " (" + "%8.4f" % mu_t_D1_std[c] + ")    " \
                                  + "%8.4f" % mu_t_D0[c]  + " (" + "%8.4f" % mu_t_D0_std[c] + ")    " \
                                  + "%8.4f" % NormDif_t[c]) 
            c += 1
            
        # Post-balance table
        print("")
        print("Means and standard deviations of t_W (post-balance)                                        ")
        print("-------------------------------------------------------------------------------------------")
        print("                            Treated (D = 1)        Control (D = 0)        Efficient (D = 1)")
        print("-------------------------------------------------------------------------------------------")
        c = 0
        for names in t_W_names:
            print(names.ljust(25) + "%8.4f" % mu_t_s[c]    + " (" + "%8.4f" % mu_t_s_std[c]   + ")    " \
                                  + "%8.4f" % mu_t_a[c]    + " (" + "%8.4f" % mu_t_a_std[c]   + ")    " \
                                  + "%8.4f" % mu_t_eff[c]  + " (" + "%8.4f" % mu_t_eff_std[c] + ")    ") 
            c += 1     
    
    # Collect/format remaining returnables and exit function
    pscore_tests = [study_test, auxiliary_test]                 # Collect p-score test results
    tilts        = np.concatenate((pi_eff, pi_s, pi_a), axis=1) # Collect three sample tilts              

    return [gamma_ast, vcov_gamma_ast, pscore_tests, tilts, exitflag]