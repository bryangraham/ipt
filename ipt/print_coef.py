# Load library dependencies
import numpy as np
import scipy as sp

# Define print_coef() function
#-----------------------------------------------------------------------------#

def print_coef(beta, vcov, var_names=None, alpha=0.05):
    
    """
    AUTHOR: Bryan S. Graham, bgraham@econ.berkeley.edu, July 2017
            (Revised September 2018)
    PYTHON 3.6
    
    This function prints out a list of variable names, coefficient estimates and standard errors
    in a unified format. Assume beta and var_names are conformably ordered
    
    INPUTS
    -------
    beta        : K vector of estimated coefficients
    vcov        : K x K estimated variance-covariance matrix (2d numpy array)
    var_names   : list of length K with variable names
    alpha       : value for 1-alpha confidence interval
    
    
    OUTPUTS
    -------
    This function returns "None". All "output" is to screen.
        
    """

    # if var_names is None then label variables X0, X1...etc.
    if not var_names:
        var_names = []
        for k in range(0,len(beta)):
            var_names.append("X_" + str(k))
            
    print("")
    print("Independent variable       Coef.    ( Std. Err.)     (" + "%0.2f" % (1-alpha) + " Confid. Interval )")
    print("-------------------------------------------------------------------------------------------")
        
    c = 0
    crit = sp.stats.norm.ppf(1-alpha/2)
    for names in var_names:
        print(names.ljust(25) + "%10.6f" % beta[c] + \
                         " (" + "%10.6f" % np.sqrt(vcov[c,c]) + ")" + \
                         "     (" + "%10.6f" % (beta[c] - crit*np.sqrt(vcov[c,c])) + \
                         " ,"     + "%10.6f" % (beta[c] + crit*np.sqrt(vcov[c,c])) + ")")
        c += 1
                
    print("")
    print("-------------------------------------------------------------------------------------------")
       
    
    return None