"""

solve.py
--------
This code solves the model.

"""

#%% Imports from Python
from numpy import argmax,expand_dims,inf,squeeze,tile,zeros,seterr
from types import SimpleNamespace
import time
seterr(divide='ignore')
seterr(invalid='ignore')

#%% Solve the model using Backward Induction.
def experience_life(myClass):
    '''
    
    This function solves the stochastic life cycle model.
    
    Input:
        myClass : Model class with parameters, grids, and utility function.
        
    '''

    print('\n--------------------------------------------------------------------------------------------------')
    print('Solving the Model by Backward Induction')
    print('--------------------------------------------------------------------------------------------------\n')
    
    # Namespace for optimal policy funtions.
    setattr(myClass,'sol',SimpleNamespace())
    sol = myClass.sol

    # Model parameters, grids and functions.
    
    par = myClass.par # Parameters.
    
    T = par.T # Last period of life.
    tr = par.tr # First year of retirement.
    
    beta = par.beta # Discount factor.
    sigma = par.sigma # CRRA.
    
    alen = par.alen # Grid size for a.
    agrid = par.agrid # Grid for a (state and choice).

    ylen = par.ylen # Grid size for y.
    ygrid = par.ygrid # Grid for y.
    pmat = par.pmat # Transition matrix for y.

    r = par.r # Real interest rate.
    kappa = par.kappa # Share of income as pension.

    util = par.util # Utility function.

    amat = tile(expand_dims(agrid,axis=1),(1,ylen)) # k for each value of A.
    ymat = tile(expand_dims(ygrid,axis=0),(alen,1)) # A for each value of k.

    # Containers.
    v1 = zeros((alen,T,ylen)) # Container for V.
    a1 = zeros((alen,T,ylen)) # Container for a'.
    c1 = zeros((alen,T,ylen)) # Container for c.

    t0 = time.time()

    for age in reversed(range(T)): # Iterate on the Bellman Equation until convergence.
    
        if age == T-1: # Python index starts at 0 and does not include the upper limit.
            
            c1[:,age,:] = amat + kappa*ymat
            a1[:,age,:] = 0.0
            v1[:,age,:] = util(c1[:,age,:],sigma)
    
        else:
            
            for i in range(ylen): # Loop over the y-states.

                if age >= tr: # Workers get a salary; retiress get a pension.
                    yt = kappa*ygrid[0][i]
                    ev = squeeze(v1[:,age+1,i])
                else:
                    yt = ygrid[0][i]
                    ev = squeeze(squeeze(v1[:,age+1,:])@pmat[i,:].T)
                    
                for p in range(0,alen): # Loop over the a-states.
                
                    # Consumption.
                    ct = agrid[p]+yt-(agrid/(1.0+r))
                    ct[ct<0.0] = 0.0
                    
                    # Solve the maximization problem.
                    vall = util(ct,sigma) + beta*ev # Compute the value function for each choice of a', given a.
                    vall[ct<=0.0] = -inf # Set the value function to negative infinity number when c <= 0.
                    v1[p,age,i] = max(vall) # Maximize: vmax is the maximized value function; ind is where it is in the grid.
                    a1[p,age,i] = agrid[argmax(vall)] # Optimal a'.
                    c1[p,age,i] = ct[argmax(vall)] # Optimal a'.
                
        # Print counter.
        if age%5 == 0:
            print('Age: ',age,'.\n')

    t1 = time.time()
    print('Elapsed time is ',t1-t0,' seconds.')

    # Macro variables, value, and policy functions.
    sol.c = c1 # Consumption policy function.
    sol.a = a1 # Saving policy function.
    sol.v = v1 # Value function.
