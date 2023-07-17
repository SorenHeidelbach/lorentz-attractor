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
    >>> lorenz.step_x(1, 2, 3)
    1.1
    >>> lorenz.simulation
    (array([0.1    , 0.1    , 0.10049]), array([0.1       , 0.1049    , 0.10975357]), array([0.1       , 0.09743333, 0.09494001]), array([0.   , 0.015, 0.03 ]))
    """
    def __init__(self, x0=0.1, y0=0.1, z0=0.1, nstep=5000, dt=0.01, sigma=10, beta=8/3, rho=6, method="euler", dtype=np.float64) -> None:
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

    def step_x(self, x: float, y: float, z: float) -> float:
        return x + self.dt * self.sigma * (y - x)
    
    def step_y(self, x: float, y: float, z: float) -> float:
        return y + self.dt * (x * (self.rho - z) - y)
    
    def step_z(self, x: float, y: float, z: float) -> float:
        return z + self.dt * (x * y - self.beta * z)
    
    def euler(self, x, y, z) -> tuple:
        x1 = self.step_x(x, y, z)
        y1 = self.step_y(x, y, z)
        z1 = self.step_z(x, y, z)
        return x1, y1, z1
    
    def solve(self):
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
        os.makedirs(prefix, exist_ok=True)
        filename = f"lorenz_attractor_{self.method.__name__}_sigma{self.sigma:.1f}_beta{self.beta:.1f}_rho{self.rho:.1f}_x0{self.x0:.1f}_y0{self.y0:.1f}_z0{self.z0:.1f}.npz"
        filepath = os.path.join(prefix, filename)
        np.savez(filepath, sim = self.simulation, allow_pickle=True)


