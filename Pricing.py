import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy as sp
import scipy.linalg as linalg 
plt.rcParams["figure.figsize"] = (16,9)
plt.style.use("dark_background")

def get_K(S_0, moneyness, call_or_put):
    """
    function to return strike price for a call or put given spot price and moneyness
    """
    if call_or_put == 'c':
        
        K = S_0/moneyness

    else: #put
        K = S_0 * moneyness
    
    return K

#~~~~~~~~~~~~~~~~~~~~~~~~____Implicit scheme for call-spread_____~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#1) Implicit method
def pde_scheme_implicit_call_spread(S_0, r, sigma, T, K1, K2, H, nb_x_side, nb_t):
    nb_x = 2 * nb_x_side + 1
    xs = np.linspace(np.log(S_0)-H, np.log(S_0)+H, nb_x)
    dx = 2. * H / (nb_x - 1.) 
    dt = T / (nb_t - 1.)
    ts = np.linspace(0, T, nb_t)
    p = np.empty([nb_x, nb_t])
    g = lambda S : (np.maximum(S-K1,0.) - np.maximum(S-K2,0.))
    #setup boundaries contidions
    p[:,0] = g(np.exp(xs))
    p[0,:] = 0.
    p[-1,:] = (K2 - K1) * np.exp(-r*ts)
    #setup coefficients
    d = 1.+dt*(r + (r-0.5*sigma**2)/dx + sigma**2 / dx**2)
    sup_d = -dt*((r-0.5*sigma**2)/dx + 0.5 * sigma**2 / dx**2)
    inf_d = -dt*(0.5 * sigma**2 / dx**2)
    #fill the matrix
    A = np.diag(d * np.ones(nb_x-2)) + np.diag(sup_d *\
        np.ones(nb_x-3), 1) + np.diag(inf_d * np.ones(nb_x-3), -1)
    v = np.zeros_like(p[1:-1,0])
    v[-1] = 1.
    invA = np.linalg.inv(A)
    for t in range(1,nb_t):
        p[1:-1,t] = invA @ (p[1:-1,t-1] - sup_d * p[-1,t] * v)   
    return p, p[nb_x_side,-1]


#~~~~~~~~~~~~~~~~~~~~~~~~~~_______Black-Scholes Formula for call spread_______~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def BSM_call_formula(S_0, K, r, sigma, T):
    d_1 = (np.log(S_0/K) + (r + sigma**2/2) * T)/(sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)
    return S_0 * norm.cdf(d_1) - K * np.exp(-r*T) * norm.cdf(d_2)

       
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~_______Crank-Nicolson PDE scheme for call spread_____~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pde_scheme_cranky_call_spread(S_0, r, sigma, T, K1, K2, H, nb_x_side, nb_t):
    nb_x = 2 * nb_x_side + 1
    xs = np.linspace(np.log(S_0)-H, np.log(S_0)+H, nb_x)
    dx = 2. * H / (nb_x - 1.) 
    dt = T / (nb_t - 1.)
    ts = np.linspace(0, T, nb_t)
    p = np.empty([nb_x, nb_t])
    q = np.empty([nb_x, nb_t])
    g = lambda S : (np.maximum(S-K1,0.) - np.maximum(S-K2,0.))
    #here the boundary condition
    p[:,0] = g(np.exp(xs))
    p[0,:] = 0.
    p[-1,:] = (K2 - K1) * np.exp(-r*ts)
    #setup coefficients - explicit scheme portion
    A_d = 1.-dt*(r + (r-0.5*sigma**2)/dx + sigma**2 / dx**2)#there is a mistake here
    A_sup_d = dt*((r-0.5*sigma**2)/dx + 0.5 * sigma**2 / dx**2) 
    A_inf_d = dt*(0.5 * sigma**2 / dx**2)         
    A = np.zeros([nb_x-2, nb_x])
    A[:,1:-1] = np.diag(A_d * np.ones(nb_x-2)) + np.diag(A_sup_d * \
                np.ones(nb_x-3), 1) + np.diag(A_inf_d * np.ones(nb_x-3), -1)
    #setup coefficients - implicit scheme portion
    B_d = 1.+dt*(r + (r-0.5*sigma**2)/dx + sigma**2 / dx**2)
    B_sup_d = -dt*((r-0.5*sigma**2)/dx + 0.5 * sigma**2 / dx**2)
    B_inf_d = -dt*(0.5 * sigma**2 / dx**2)
    #solve the matrix system
    B = np.diag(B_d * np.ones(nb_x-2)) + np.diag(B_sup_d * \
        np.ones(nb_x-3), 1) + np.diag(B_inf_d * np.ones(nb_x-3), -1)
    invB = np.linalg.inv(B)
    # combining coefficients 
    A[0,0] = A_inf_d - B_inf_d
    A[-1,-1] = A_sup_d - B_sup_d
    
    for t in range(1,nb_t):
        p[1:-1,t] = invB @ (A @ p[:,t-1]) 
    return p,p[nb_x_side,-1]


#~~~~~~~~~~~~~~~~~~~_____Implicit scheme for put option____~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pde_scheme_implicit_put(S_0, r, sigma, T, K, H, nb_x_side, nb_t):
    nb_x = 2 * nb_x_side + 1
    xs = np.linspace(np.log(S_0)-H, np.log(S_0)+H, nb_x)
    dx = 2. * H / (nb_x - 1.) 
    dt = T / (nb_t - 1.)
    ts = np.linspace(0, T, nb_t)
    p = np.empty([nb_x, nb_t])
    g = lambda S : np.maximum(K-S,0.)
    #setup boundaries conditions
    p[:,0] = g(np.exp(xs))
    p[0,:] = K * np.exp(-r*ts) - np.exp(np.log(S_0)-H)
    p[-1,:] = 0.
    #setup coefficients
    d = 1.+dt*(r + (r-0.5*sigma**2)/dx + sigma**2 / dx**2)
    sup_d = -dt*((r-0.5*sigma**2)/dx + 0.5 * sigma**2 / dx**2)
    inf_d = -dt*(0.5 * sigma**2 / dx**2)
    A = np.diag(d * np.ones(nb_x-2)) + np.diag(sup_d *\
        np.ones(nb_x-3), 1) + np.diag(inf_d * np.ones(nb_x-3), -1)
    v = np.zeros_like(p[1:-1,0])
    v[0] = 1.
    invA = np.linalg.inv(A)
    for t in range(1,nb_t):
        p[1:-1,t] = invA @ (p[1:-1,t-1] - inf_d * p[0,t] * v)   
    return p, p[nb_x_side,-1]
#~~~~~~~~~~~~~~~~~~~~~~~~~____Black-Scholes formula for put_____~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def BSM_put_formula(S_0, K, r, sigma, T):
    d_1 = (np.log(S_0/K) + (r + sigma**2/2) * T)/(sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d_2) - S_0 * norm.cdf(-d_1)

#~~~~~~~~~~~~~~~~~~~~~____Crank-Nicolson scheme for put______~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pde_scheme_cranky_put(S_0, r, sigma, T, K, H, nb_x_side, nb_t):
    nb_x = 2 * nb_x_side + 1
    xs = np.linspace(np.log(S_0)-H, np.log(S_0)+H, nb_x)
    dx = 2. * H / (nb_x - 1.) 
    dt = T / (nb_t - 1.)
    ts = np.linspace(0, T, nb_t)
    p = np.empty([nb_x, nb_t])
    q = np.empty([nb_x, nb_t])
    g = lambda S : np.maximum(K-S,0.)
    #Boundaries conditions
    p[:,0] = g(np.exp(xs))
    p[0,:] = K * np.exp(-r*ts) - np.exp(np.log(S_0)-H)
    p[-1,:] = 0.
    #Coefficients in the matrix
    A_d = 1.-dt*(r + (r-0.5*sigma**2)/dx + sigma**2 / dx**2)
    A_sup_d = dt*((r-0.5*sigma**2)/dx + 0.5 * sigma**2 / dx**2) 
    A_inf_d = dt*(0.5 * sigma**2 / dx**2)         
    A = np.zeros([nb_x-2, nb_x])
    A[:,1:-1] = np.diag(A_d * np.ones(nb_x-2)) + np.diag(A_sup_d *\
                np.ones(nb_x-3), 1) + np.diag(A_inf_d * np.ones(nb_x-3), -1)

    B_d = 1.+dt*(r + (r-0.5*sigma**2)/dx + sigma**2 / dx**2)
    B_sup_d = -dt*((r-0.5*sigma**2)/dx + 0.5 * sigma**2 / dx**2)
    B_inf_d = -dt*(0.5 * sigma**2 / dx**2)
    B = np.diag(B_d * np.ones(nb_x-2)) + np.diag(B_sup_d * \
        np.ones(nb_x-3), 1) + np.diag(B_inf_d * np.ones(nb_x-3), -1)
    invB = np.linalg.inv(B)
    
    A[0,0] = A_inf_d - B_inf_d
    A[-1,-1] = A_sup_d -B_sup_d
    #Solve the matrix
    for t in range(1,nb_t):
        p[1:-1,t] = invB @ (A @ p[:,t-1]) 
    return p,p[nb_x_side,-1]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~____Plot the graph_____~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def graph(table ,S_0 ,H ,nb_x):
    xs = np.linspace(np.log(S_0)-H, np.log(S_0)+H, nb_x)
    plt.plot(np.exp(xs),table[:,0],'b', label = 'payoff')
    plt.plot(np.exp(xs), table[:,-1],'r', label = 'price')
    plt.legend()
    return plt.show()
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def setup_graph(Implicit, CrankNic, S_0, H, nb_x):
    xs = np.linspace(np.log(S_0)-H, np.log(S_0)+H, nb_x)
    plt.plot(np.exp(xs),Implicit[:,0],'b', label = 'Payoff')
    plt.plot(np.exp(xs), Implicit[:,-1],'r', label = 'Implicit')
    plt.plot(np.exp(xs), CrankNic[:,-1],'y', label = 'Crank nicolson')
    plt.legend()
    return plt.show()
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~____End_____~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




