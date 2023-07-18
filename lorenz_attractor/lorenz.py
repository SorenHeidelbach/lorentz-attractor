import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit
from colour import Color
import os


class LorenzAttractor:
    """
    A class to simulate the Lorenz Attractor system, a system of differential equations.

    This class uses two different methods for the simulation, Euler and Runge-Kutta of order 4.
    
    Parameters
    ----------
    x0 : float
        initial x value.
    y0 : float
        initial y value.
    z0 : float
        initial z value.
    nstep : int
        the number of steps for the simulation.
    dt : float
        time step size.
    sigma : float
        sigma parameter for Lorenz system.
    beta : float
        beta parameter for Lorenz system.
    rho : float
        rho parameter for Lorenz system.
    method : str
        the method to be used for simulation, either 'euler'.

    Examples
    -------
    >>> lorenz = LorenzAttractor(nstep=3)
    >>> lorenz.solve()
    """
    def __init__(self, x0=0.1, y0=0.1, z0=0.1, nstep=5000, dt=0.01, sigma=10, beta=8/3, rho=6, method="euler", dtype=np.float64) -> None:
        """
        Initializes the Lorenz Attractor object with initial conditions and parameters.
        """
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.nstep = nstep
        self.dt = dt
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.simulation = None
        self.dtype = dtype
        self.allowed_method = {
            "euler": self.euler
        }
        assert method in self.allowed_method.keys(), f"method must be one of {self.allowed_method.keys()}"
        self.define_method(method)

    def define_method(self, method):
        self.method = self.allowed_method[method]
        """
        Defines the method to be used for the Lorenz system simulation.

        Parameters
        ----------
        method : str
            The method to be used for simulation. Must be one of the keys in allowed_method.
        """

    def step_x(self, x: float, y: float) -> float:
        """
        Calculates the next x value in the Lorenz system.

        Parameters
        ----------
        x : float
            Current x value.
        y : float
            Current y value.

        Returns
        -------
        float
            Next x

        Examples
        --------
        >>> lorenz = LorenzAttractor(sigma = 10, dt = 0.1)
        >>> lorenz.step_x(1, 2)
        2.0
        """
        return x + self.dt * self.sigma * (y - x)
    
    def step_y(self, x: float, y: float, z: float) -> float:
        """
        Calculates the next y value in the Lorenz system.

        Parameters
        ----------
        x : float
            Current x value.
        y : float
            Current y value.
        z : float
            Current z value.

        Returns
        -------
        float
            Next y

        Examples
        --------
        >>> lorenz = LorenzAttractor(rho = 4, dt = 1.)
        >>> lorenz.step_y(2, 1, 3)
        2.0
        """
        return y + self.dt * (x * (self.rho - z) - y)
    
    def step_z(self, x: float, y: float, z: float) -> float:
        """
        Calculates the next z value in the Lorenz system.

        Parameters
        ----------
        x : float
            Current x value.
        y : float
            Current y value.
        z : float
            Current z value.

        Returns
        -------
        float
            Next z

        Examples
        --------
        >>> lorenz = LorenzAttractor(beta = 1, dt = 1.)
        >>> lorenz.step_z(1, 2, 1)
        2.0
        """
        return z + self.dt * (x * y - self.beta * z)
    
    def euler(self, x, y, z) -> tuple:
        """
        Calculates the next x, y, and z values in the Lorenz system using Euler's method.

        Parameters
        ----------
        x : float
            Current x value.
        y : float
            Current y value.
        z : float
            Current z value.

        Returns
        -------
        tuple
            Next x, y, and z values.

        Examples
        --------
        >>> lorenz = LorenzAttractor(sigma = 10, rho = 4, beta = 1, dt = 1.)
        >>> lorenz.euler(1, 2, 3)
        (11.0, 1.0, 2.0)
        """
        x1 = self.step_x(x, y)
        y1 = self.step_y(x, y, z)
        z1 = self.step_z(x, y, z)
        return x1, y1, z1
    
    def solve(self):
        """
        Solves the Lorenz system using the defined method and stores the result in self.simulation.

        Examples
        --------
        >>> lorenz = LorenzAttractor(nstep=3)
        >>> lorenz.solve()
        >>> lorenz.simulation
        (array([0.1    , 0.1    , 0.10049]), array([0.1       , 0.1049    , 0.10975357]), array([0.1       , 0.09743333, 0.09494001]), array([0.   , 0.015, 0.03 ]))
        """
        x = np.zeros(self.nstep, dtype=self.dtype)
        y = np.zeros(self.nstep, dtype=self.dtype)
        z = np.zeros(self.nstep, dtype=self.dtype)
        t = np.linspace(0, self.nstep * self.dt, self.nstep)
        x[0] = self.x0
        y[0] = self.y0
        z[0] = self.z0

        for i in range(self.nstep - 1):
            x[i+1], y[i+1], z[i+1] = self.method(x[i], y[i], z[i])
        self.simulation = x, y, z, t
    
    def save_simulation(self, prefix=""):
        """
        Saves the simulation results to a file.

        Parameters
        ----------
        prefix : str, optional
            Directory in which to save the file. The default is "", current working directionary.

        Examples
        --------
        >>> lorenz = LorenzAttractor(nstep=3)
        >>> lorenz.solve()
        >>> lorenz.save_simulation(prefix="/tmp")
        """
        os.makedirs(prefix, exist_ok=True)
        filename = f"lorenz_attractor_{self.method.__name__}_sigma{self.sigma:.1f}_beta{self.beta:.1f}_rho{self.rho:.1f}_x0{self.x0:.1f}_y0{self.y0:.1f}_z0{self.z0:.1f}.npz"
        filepath = os.path.join(prefix, filename)
        np.savez(filepath, sim = self.simulation, allow_pickle=True)


