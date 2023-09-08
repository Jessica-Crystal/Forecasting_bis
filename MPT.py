#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Import Library~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


############################################################################################################################
#Equally Weighted Portfolio

def weight_EWP(r):
    """ compute the weight of equally weight portfolio (EWP) for given return parameter r,
    each component is equal to 1/len(r) and the sum is equal to 1"""
    d = len(r.columns)
    return pd.Series(1/d, index = r.columns)

def portfolio_ret(r):#computes the portfolio return of the equally weight
    """Return the portfolio of equally weight for given return parameter r"""
    return (r*weight_EWP(r)).sum(axis = 1)

###########################################################################################################################
#Markowitz Minimum Variance

def weights(r):
    '''gererate a random weight'''
    weights = np.random.random(size = len(r.columns))
    return weights/sum(weights)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Var_matrix(W, cov_mat):
    """
    Computes the function in the equation (*) from a covariance matrix and constituent weights
    weights are a numpy array or d x 1 matrix and cov_matrix is an d x d matrix.
    """
    return W.T @ cov_mat@ W

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def minimize_variance(cov_mat):
    """
    compute the optimal weights that achieve the target return
    given a covariance matrix
    """
    d = cov_mat.shape[0]
    bounds = ((0.0, 1.0),) * d
    init_guess = np.repeat(1/d, d)
    # construct the constraints
    constraints = {'type': 'eq','fun': lambda Weights: np.sum(Weights) - 1}
   
    Weights = minimize(Var_matrix, init_guess,
                       args = (cov_mat,), method='SLSQP',
                       options={'disp': False},
                       constraints=constraints,
                       bounds=bounds)
    return Weights.x

######################################################################################################################
def sigma(w, covmat):
    """
    Computes the volalitity of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or d x 1 maxtrix and covmat is an d x d matrix
    """
    vol = np.sqrt(w.T @ covmat @ w)
    return vol 

def sample_cov(returns, **kwargs):
    return returns.cov()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def risk_contribution(Weights,cov_matrix):
    """
    Compute the contributions to risk of the constituents of a portfolio,
    given a set of portfolio weights and a covariance matrix
    """
    total_portfolio_var = sigma(Weights,cov_matrix)**2
    # Marginal contribution of each component
    marginal_contrib = cov_matrix@Weights
    risk_contrib = np.multiply(marginal_contrib,Weights.T)/total_portfolio_var
    return risk_contrib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def target_risk_contributions(target_risk, cov_matrix):
    """
    Compute the weights of the portfolio that gives the weights such
    that the contributions to portfolio risk are as close as possible to
    the target_risk, given the covariance matrix
    """
    d = cov_matrix.shape[0]
    init_guess = np.repeat(1/d, d)
    bounds = ((0.0, 1.0),) * d # an 10 times of 2-tuples
    # build a constraint the constraints
    constraints = {'type': 'eq',
                        'fun': lambda Weights: np.sum(Weights) - 1
    }
    def msd_risk(Weights, target_risk, cov_matrix):
        """
        Compute the Mean Squared Difference in risk contributions
        between weights and target_risk
        """
        w_contribs = risk_contribution(Weights, cov_matrix)
        return ((w_contribs-target_risk)**2).sum()
    
    Weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov_matrix), method='SLSQP',
                       options={'disp': False},
                       constraints=(constraints,),
                       bounds=bounds)
    return Weights.x
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def erc_portfolio(cov_matrix):
    """
    Returns the weights of the portfolio that equalizes the contributions
    of the constituents based on the given covariance matrix
    """
    n = cov_matrix.shape[0]
    return target_risk_contributions(target_risk=np.repeat(1/n,n), cov_matrix=cov_matrix)

def weight_erc(returns, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the ERC portfolio given a covariance matrix of the returns 
    """
    est_cov = cov_estimator(returns, **kwargs)
    return erc_portfolio(est_cov)   
#################################################################################################################
