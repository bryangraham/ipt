ipt: a Python 3.7 package for causal inference by inverse probability tilting
-----------------------------------------------------------------------------
by Bryan S. Graham, UC - Berkeley, e-mail: bgraham@econ.berkeley.edu

This package includes a Python 3.6 implementation of the Average Treatment Effect of the 
Treated (ATT) estimator introduced in Graham, Pinto and Egel (2016). The function att() 
allows for sampling weights as well as "clustered standard errors", but these features 
have not yet been extensively tested.

The package also includes a particular implementation of the E-estimator for the
partially linear regression model due to Newey (1990) and Robins, Mark and Newey (1992).

An implementation of the Average Treatment Effect (ATE) estimator introduced in Graham, 
Pinto and Egel (2012) is planned for a future update (as well as other causal
inference estimation procedures).

This package is offered "as is", without warranty, implicit or otherwise. While I would
appreciate bug reports, suggestions for improvements and so on, I am unable to provide any
meaningful user-support. Please e-mail me at bgraham@econ.berkeley.edu

Please cite both the code and the underlying source articles listed below when using this 
code in your research.


CODE CITATION
---------------
Graham, Bryan S. (2017). "ipt: a Python 3.7 package for causal inference by inverse 
	probability tilting," (Version 0.2.2) [Computer program]. Available at 
	https://github.com/bryangraham/ipt (Accessed 04 Oct 2018) 
	
PAPER CITATIONS
---------------
Graham, Bryan S., Cristine Pinto and Daniel Egel. (2012). “Inverse probability tilting 
	for moment condition models with missing data,” Review of Economic Studies 79 (3): 
	1053 - 1079

Graham, Bryan S., Cristine Pinto and Daniel Egel. (2016). “Efficient estimation of data 
	combination models by the  method of auxiliary-to-study tilting (AST),” Journal of 
	Business and Economic Statistics 31 (2): 288 - 301 	

Newey, Whitney. (1990). "Semiparametric efficiency bounds," Journal of Applied 
	Econometrics 5 (2):	99 - 135
	
Robins, James M., Mark, Steven D. and Newey, Whitney K. (1992). "Estimating exposure 
	effects by modelling the expectation of exposure conditional on confounders,"
	Biometrics 48 (2): 479 - 495