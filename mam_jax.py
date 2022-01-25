import numpy as np
import pickle
from scipy.optimize import minimize

class MamJax:
    """
    Class for running of the MAM algorithm with JAX. 
    
    - Uses scipy.minimise to implement L-BFGS-B method.
    - Uses JAX.jacfwd via JFW object to calculate the jacobian of the FW action.
    - User should set bounds for optimisation problem via .bnds property.
    
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
        User provides shape of (time, ndim, 2).Indexed by: Time, ndim, lower/upper bound.
        Class reshapes for scipy.minimise usage.
        
    Methods
    -----------
    run(options)
        Runs MAM algorithm until convergence or user specified limit.
    
    Attributes
    -----------
    
    res: scipy.optimize.OptimizeResult
        Latest result from scipy minimise.
        See scipy docs for the info it contains.
    
    p: list 
        Arguments used for drift and diffusion functions in the JFW object.
        
    ic: np array
        Initial condition for minimisation problem. 
        Shape is (time, ndim). 

    time: np array
        Time points that paths are parameterised along
        Shape (time)
        
    nit: integer
        Number of algorithm iterations thus far.

    """
    
    def __init__(self, JFW, ic, time, p, bnds=None):
        """
        Parameters
        ----------
        JFW: JFW Object 
            Object that has ability to calculate FW action and it's jacobian.
            Does most of the heavy lifting.
    
        ic: np array
            Initial instanton path. 
            Shape is (time, ndim). 
            
        time: np array
            Time points that paths are parameterised along.
            Shape (time).
            
        p: list 
            Arguments for drift and diffusion functions
            
        """
        
        # User Accessed attrbutes
        self.time = time
        self.p = p
        self.ic = ic
        self.nit=0
        
        # Internal attributes
        self._JFW = JFW
        self._user_shape = ic.shape
        self._ic = ic.flatten()
        self._instanton = ic.flatten() # Current state of minimisation

        # Create default bounds if user doesn't set
        if (bnds is None):
            self._initialise_bnds() 
        else:
            self.bnds = bnds
        
    def _action(self, flat_path):
        path = flat_path.reshape(self._user_shape)      
        return self._JFW.action(path, self.time, self.p)
    
    def _jacobian(self, flat_path):
        path = flat_path.reshape(self._user_shape)      
        return self._JFW.jacobian(path, self.time, self.p).flatten()
    
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
        
    def run(self, opt):
        "Runs the MAM algorithm. opt feeds scipy.minimize via options argument."
        res = minimize(self._action, self._instanton, method='L-BFGS-B', bounds = self._bnds, 
                   jac=self._jacobian, options=opt)
        
        # Update current info
        self._instanton = res.x
        self.nit += res.nit
        self.res = res 
        return 
    
    def save(self, name):
        info = {
            'Result': self.res,
            'Parameters': self.p,
            'nit': self.nit,
            'time': self.time
                        }
        with open(f'{name}.pickle','wb') as file:
            pickle.dump(info, file)
        print(f'Saved at {name}.pickle')
        
        