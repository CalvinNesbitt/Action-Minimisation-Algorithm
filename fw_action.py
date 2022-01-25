""" 
----------
Contents
----------
- FreidlinWentzell, class for calculating the FW action of a path.

- JFW, child class of FreidlinWentzell that also calculates the jacobian of the FW Action.
"""

# ----------------------------------------
# Dependicies
# ----------------------------------------
import numpy as np
import jax
from joblib import Parallel, delayed
import multiprocessing

# ----------------------------------------
# FreidlinWentzell
# ----------------------------------------

class FreidlinWentzell:
    
    """
    Calculates the FW action of a path.
    
    - __init__ requires knowledge of the drift and diffusion.
    - The .action(path, time, p) method returns FW action for different paths.
    - Velocity is approximated using finite differences.
    - Integration done via trapezoidal rule.
            
    Methods
    -----------
    action(path, time, p)
        Calculates the FW action of a given path.
    """
    
    def __init__(self, b, diff_inv):
        """
        Parameters
        ----------
        b: function 
            Takes point on path and returns drift at point.
            Path input and are drift output are shape (ndim).
            Should be of form b(point, args).
            
        diff_inv: function
            Takes point along the path and returns inverse of diffusion matrix.
            Point on path input is of shape (ndim).
            Should be of form D_inv(point, args).
            Return is matrix of shape (ndim, ndim).
        """
        
        self._b = b
        self._diff_inv = diff_inv
    
    ########################################################
    ## Defining Lagrangians at points along the path.
    ## Need to used different differencing at end points
    ########################################################
    
    # Forward differencing for initial velocity
    # x should be shape (2, ndim)
    
    def _v0(self, x, dt, p):
        return (x[1] - x[0])/dt - self._b(x[0], p)

    def _L0(self, x, dt, p):
        v = self._v0(x, dt, p)
        return  v @ self._diff_inv(x[0], p) @ v

    # Cenetered differencing away from the boundary points
    # x should be shape (3, ndim)
    
    def _vk(self, x, dt, p):
        return (x[2] - x[0])/(2*dt) - self._b(x[1], p)

    def _Lk(self, x, dt, p):
        v = self._vk(x, dt, p)
        return v @ self._diff_inv(x[1], p) @ v

    # Backward differencing for final velocity
    # x should be shape (2, ndim)
    
    def _vN(self, x, dt, p):
        return (x[1] - x[0])/dt - self._b(x[1], p)

    def _LN(self, x, dt, p):
        v = self._vN(x, dt, p)
        return v @ self._diff_inv(x[1], p) @ v 
    
    def action(self, path, time, p):
        """
        Calculates the FW action of a given path.
        
        Parameters
        ----------
        path: np array
            The path we are calculating the FW action of.
            Expected shape is (time, ndim).
        time: np array
            The time parameterising the path.
        p: list
            Arguments for drift and diffusion functions.
        """
        Ls = np.empty(path.shape[0])

        #Boundary Values
        Ls[0] = self._L0(path[:2], time[1] - time[0], p)
        Ls[-1] = self._LN(path[-2:], time[-1] - time[-2], p)

        # Middle Values
        for i in range(1, path.shape[0] - 1):
            Ls[i] = self._Lk(path[i-1:i+2], time[i+1] - time[i], p)
        return 0.5 * np.trapz(Ls, dx=time[1] - time[0]) 
    
    def lagrangians(self, path, time, p):
        """
        Calculates the Lagrangians along a given path. Useful for action integral plots
        
        Parameters
        ----------
        path: np array
            The path we are calculating the FW action of.
            Expected shape is (time, ndim).
        time: np array
            The time parameterising the path.
        p: list
            Arguments for drift and diffusion functions.
        """
        Ls = np.empty(path.shape[0])

        #Boundary Values
        Ls[0] = self._L0(path[:2], time[1] - time[0], p)
        Ls[-1] = self._LN(path[-2:], time[-1] - time[-2], p)

        # Middle Values
        for i in range(1, path.shape[0] - 1):
            Ls[i] = self._Lk(path[i-1:i+2], time[i+1] - time[i], p)
        return Ls
    

    
    
# ----------------------------------------
# JFW
# ----------------------------------------
    
class JFW(FreidlinWentzell):
    """
    Extends the FreidlinWentzell class to also calculate the jacobian of the FW action.
    
    - __init__ requires knowledge of the drift and diffusion (see FreidlinWentzell doc).
    - The .jacobian(path, time, p) method returns jacobian of the FW action at a particular path.
    - Uses JAX to calculate the jacobian.
            
    Methods
    -----------
    action(path, time, p)
        Calculates the FW action of a given path.

    jacobian(path, time, p)
        Calculates the jacobian of the FW action at a given path.
    """
    
    def __init__(self, b, diff_inv):
        super().__init__(b, diff_inv)
        
        # JAX calculating the derivatives we want
        self._grad_L0 = jax.jacfwd(self._L0)
        self._grad_Lk = jax.jacfwd(self._Lk)
        self._grad_LN = jax.jacfwd(self._LN)

    def jacobian(self, path, time, p):
        """
        Calculates the jacobian of the FW action at a given path.
        Relies on jax and multiprocessing.
        
        Parameters
        ----------
        path: np array
            The path we are evaluating the jacobian at.
            Expected shape is (time, ndim).
        time: np array
            The time parameterising the path.
        p: list
            Arguments for drift and diffusion functions in FW action.
        """
        dt = time[1] - time[0] # Assumed constant time spacing

        grad_Ls = np.empty(path.shape)

        # Boundary Values - dt/2 from trapz rule
        grad_Ls[0] = dt/2 * self._grad_L0(path[:2], dt, p)[0] + dt * self._grad_Lk(path[:3], dt, p)[0]
        grad_Ls[1] = dt/2 * self._grad_L0(path[:2], dt, p)[1] + dt * (self._grad_Lk(path[:3], dt, p)[1] + self._grad_Lk(path[1:4], dt, p)[0])

        grad_Ls[-2] =(
            dt/2 * self._grad_LN(path[-2:], dt, p)[0] + dt * (self._grad_Lk(path[-3:], dt, p)[1] + self._grad_Lk(path[-4:-1], dt, p)[2])
        )
        grad_Ls[-1] = dt/2 * self._grad_LN(path[-2:], dt, p)[1] + dt * self._grad_Lk(path[-3:], dt, p)[2]

        # Computing grad_Lk[phi_k] in parallel
        num_cores = 32#multiprocessing.cpu_count()
        dv_list = Parallel(n_jobs=num_cores)(delayed(self._grad_Lk)(path[i-2:i+1], dt, p) for i in range(2, path.shape[0]))

        # Summing the three terms in each grad_L derivative
        for i in range(len(dv_list[:-2])):
            grad_Ls[i+2] = dt * (dv_list[i][2] + dv_list[i+1][1] + dv_list[i+2][0])
        return 0.5 * grad_Ls
