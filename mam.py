import numpy as np
from scipy.optimize import minimize

class Mam_alg:
    """
    Class for an convient run of the MAM algorithm. Built on scipy.minimise method.
    
    - Object creation requires drift and initial path (see __init__).
    - Once object is created, user should set bounds for optimisation problem via .bnds property.
    - Use of .run(options) method then evolves .instanton property under MAM algorithm.
    - .instanton property contains current state of minmisation.
    
    Properties
    -----------
    instanton: np array
        Current path being evolved by MAM, should converge to instanton. 
        User shape is (time, ndim).
        Class reshapes for scipy.minimise usage.
        
    bnds: np array
        Constraints for the minimisation problem.
        Should be set before running. Object performs default initialisation.
        User must set all bnds at the same time.
        User provides shape of (T, ndim, 2).Indexed by: Time, ndim, lower/upper bound.
        Class reshapes for scipy.minimise usage.
        
    Methods
    -----------
    run(options)
        Runs MAM algorithm until convergence or user specified limit.
    
    Attributes
    -----------
    
    b_args: list 
        Arguments used for drift b.
        Note b is of form b(path, b_args)

    ic: np array
        Initial condition for minimisation problem. 
        Shape is (time, ndim). 
        Important that initial and final point specified.

    time: np array
        Time points that paths are parameterised along
        Shape (T)
        
    nfev: integer
        Number of function evaluations in MAM thus far.

    """
    
    def __init__(self, b, b_args, d, ic, time, bnds=None):
        """
        Parameters
        ----------
        b: function 
            Takes path and returns drift.
            Path input and are drift output are shape (time, ndim).
            Should be of form b(path, b_args).
            
        b_args: list 
            Further arguments for b.
            Note b is of form b(path, b_args)
            
        d: function
            Takes point along the path and returns inverse of diffusion matrix.
            Of form d(point, b_args).
            Point on path input is of shape (ndim).
            Return is matrix of shape (ndim, ndim).
    
        ic: np array
            Initial instanton path. 
            Shape is (time, ndim). 
            Important that initial and final point specified.
            
        time: np array
            Time points that paths are parameterised along
            Shape (T)
            

        """
        
        # User Accessed attrbutes
        self.time = time
        self.b_args = b_args
        self.ic = ic
        self.nfev=0
        
        # Internal attributes
        self._b = b
        self._d = d
        self._user_shape = ic.shape
        self._ic = ic.flatten()
        self._instanton = ic.flatten() # Current state of minimisation

        # Create default bounds if user doesn't set
        if (bnds is None):
            self._initialise_bnds() 
        else:
            self.bnds = bnds
        
    def _action(self, path):
        """ Takes a path and computes the FW action along the path when ndim >1. If ndim=1,
        object will use _1d_action.

        Path input is of form necesseary for scipy.minimise (see parameter explanation).
        Finite differences are used to approximate derivatives.
        Trapezoidal Rule used to compute integral

        Parameters
        ----------
        path: np array
            shape is flat (time * ndim) for use by scipy.minimise
            method reshapes into (time, ndim)
        """
        h = path.reshape(self._user_shape)                     
        v = np.vstack(np.gradient(h, self.time, axis=0)) - self._b(h, self.b_args) #v from <v, D^-1v>

        # Dot product calulcation
        v2 = [] 
        for (x, where) in zip(v, h): #<v, D^-1v> along the path
            Dinv = self._d(where, self.b_args)
            v2.append(x.dot(Dinv @ x.T))
        return 0.5 * np.trapz(v2, x=self.time)
    
    def _1d_action(self, path):
        """
        Same as _action method for 1d case. DOESN'T YET WORK FOR STATE DEPENDENT NOISE
        """
        h = path.reshape(self._user_shape)    
        v = np.gradient(h, self.time) -  self._b(h, self.b_args)
        integrand =  v**2
        return 0.5 * np.trapz(integrand, x=self.time)
    
    @property
    def instanton(self):
        "Current state of minimisation."
        return self._instanton.reshape(self._user_shape)
    
    @property
    def bnds(self): # indexed by Time, ndim, lower/upper bound
        "Bounds of optimisation problem"
        return self._bnds.reshape((*self._user_shape, 2)) 
    
    @bnds.setter 
    def bnds(self, b): 
        # reshapes for scipy.minimise usage
        #b should be all of bounds when setting
        self._bnds = b.reshape(len(self.ic.flatten()), 2)
    
    def _initialise_bnds(self):
        """ Initialises bounds for constraint, user is free to change.
        The user specified bnds shape is different to the one used by underlying code.
        """
        
        # We initialise by creating an np array indexed by Time, ndim, lower/upper bound
        # User specified bnds are of this form
        self._bnds = np.zeros((*self._user_shape, 2))
        initial_point = self.ic[0]
        final_point = self.ic[-1]

        # Check if we have 1d case
        if (len(self._user_shape) == 1):
            shape = 1
        else:
            shape = np.ones(self._user_shape[1])

        # t = 0 Constraint
        self._bnds[0,...,0] = initial_point - 0.001 * shape # Lower bound
        self._bnds[0,...,1] = initial_point + 0.001 * shape # Upper bound

        # Bounds for t \in (dt, T-dt)
        self._bnds[1:-1, ..., 0] = - np.inf 
        self._bnds[1:-1, ..., 1] = np.inf 

        # t = T Constraint
        self._bnds[-1,...,0] = final_point - 0.001 * shape
        self._bnds[-1,...,1] = final_point + 0.001 * shape

        # We then reshape to the form used by the object
        self._bnds = self._bnds.reshape(len(self.ic.flatten()), 2)
        
    def run(self, opt, jac=None):
        "Runs the MAM algorithm. opt feds scipy.minimize via options argument."
        if (len(self._user_shape) == 1): #ndim=1 case
            res = minimize(self._1d_action, self._instanton, method='L-BFGS-B', bounds = self._bnds, 
               jac=jac, options=opt)
        else:
            res = minimize(self._action, self._instanton, method='L-BFGS-B', bounds = self._bnds, 
                   jac=jac, options=opt)
        print(res.message)
        self._instanton = res.x
        self.nfev += res.nfev
        return res
        
        