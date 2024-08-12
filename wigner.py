import numpy as np
from scipy.special import genlaguerre, factorial
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


hbar = 1
def wig_mn(m, n, x, p):
    """Wigner function of |m><n| state
    """
    if n > m:
        return wig_mn(n, m, x, -p)
        #m, n = n, m
        #p = -p

    x /= np.sqrt(hbar)
    p /= np.sqrt(hbar)
    
    return (-1)**n * (x-p*1j)**(m-n) * 1/(np.pi) * np.exp(-x*x - p*p) * \
            np.sqrt(2**(m-n) * factorial(n) / factorial(m)) * \
            genlaguerre(n, m-n)(2*x*x + 2*p*p)

def rho_mn(m,n, W, x, p):
    X,P=np.meshgrid(x,p)
    Wmn = wig_mn(m,n,X,P)
    dx = np.diff(x)[0]
    dp = np.diff(p)[0]
    
    return np.pi*2*np.sum(Wmn*W)*dx*dp

    
def rho_to_wig(rho, x, p, nmax): #optimize this

    W = np.zeros((len(x),len(p)), dtype = complex)

    X,P = np.meshgrid(x,p)
    
    for i in range(nmax):
        for j in range(nmax):
            if np.abs(rho[i,j]) > 1e-6 : 
               W += rho[i,j]*wig_mn(i,j,X,P)

    return W


def plot_wigner(ax, W, x, p):

    scale = np.max(W.real)
    nrm = Normalize(-scale, scale)

    im = ax.contourf(x,x, W.real, 100, cmap = cm.RdBu, norm =nrm)
    #plt.aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_aspect('equal')
    return im
            