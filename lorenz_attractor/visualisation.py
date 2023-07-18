import matplotlib.pyplot as plt
from matplotlib import cm
from lorenz_attractor.lorenz import LorenzAttractor
import numpy as np
from matplotlib.colors import is_color_like

def plot_simulation(lorentz_attractor, ax=None):
    """
    Plots the simulation results from a Lorenz Attractor.

    This function generates a 3D plot of the Lorenz Attractor's trajectory over time. 

    Parameters
    ----------
    lorenz_attractor : LorenzAttractor
        An instance of the LorenzAttractor class that has already run a simulation.
    ax : matplotlib.pyplot.Axes, optional
        An existing matplotlib Axes object to draw the plot onto, if any. If not provided, a new Axes object will be created.

    Returns
    -------
    None
        The function directly plots the simulation and does not return a value.

    Examples
    --------
    >>> lorentz = LorenzAttractor(nstep=10)
    >>> lorentz.solve()
    >>> plot_simulation(lorentz)
    """
    if ax == None:
        ax = plt.figure().add_subplot(projection='3d')
    x, y, z, _ = lorentz_attractor.simulation
    ax.scatter(x, y, z, marker = ".")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Lorenz Attractor, method={lorentz_attractor.method.__name__}\n sigma={lorentz_attractor.sigma:.2f}, beta={lorentz_attractor.beta:.2f}, rho={lorentz_attractor.rho:.2f}\nx0={lorentz_attractor.x0:.2f}, y0={lorentz_attractor.y0:.2f}, z0={lorentz_attractor.z0:.2f}", fontsize=10)

def plot_simulation_color(lorentz_attractor, ax=None):
    """
    Plots the simulation results from a Lorenz Attractor colored by elapsed time.

    This function generates a 3D plot of the Lorenz Attractor's trajectory over time. 

    Parameters
    ----------
    lorenz_attractor : LorenzAttractor
        An instance of the LorenzAttractor class that has already run a simulation.
    ax : matplotlib.pyplot.Axes, optional
        An existing matplotlib Axes object to draw the plot onto, if any. If not provided, a new Axes object will be created.

    Returns
    -------
    None
        The function directly plots the simulation and does not return a value.

    Examples
    --------
    >>> lorentz = LorenzAttractor(nstep=10)
    >>> lorentz.solve()
    >>> plot_simulation(lorentz)
    """
    if ax == None:
        ax = plt.add_subplot(projection='3d')
    x, y, z, t = lorentz_attractor.simulation
    colors = cm.terrain(t / max(t))
    ax.scatter(x, y, z, c=colors, marker = ".")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Lorenz Attractor, method={lorentz_attractor.method.__name__}\n sigma={lorentz_attractor.sigma:.2f}, beta={lorentz_attractor.beta:.2f}, rho={lorentz_attractor.rho:.2f}\ninital: x={lorentz_attractor.x0:.2f}, y={lorentz_attractor.y0:.2f}, z={lorentz_attractor.z0:.2f}", fontsize=10)

def plot_simulation_color_2d(lorentz_attractor, x_axis="x", y_axis="y", color="t", ax=None, size=5, label=""):
    """
    Plots a 2D scatter plot of the Lorenz Attractor simulation results.

    This function generates a 2D scatter plot of the Lorenz Attractor's trajectory over time, 
    using two specified axes.

    Parameters
    ----------
    lorenz_attractor : LorenzAttractor
        An instance of the LorenzAttractor class that has already run a simulation.
    x_axis : str, optional
        The variable to be used for the x-axis. It can be either "x", "y", "z", or "t". If it is not one of these, it should be a single number.
    y_axis : str, optional
        The variable to be used for the y-axis. It can be either "x", "y", "z", or "t". If it is not one of these, it should be a single number.
    color : str, optional
        The variable to determine the color of the dots. It can be either "x", "y", "z", or "t". If it is not one of these, it should be a valid color.
    ax : matplotlib.pyplot.Axes, optional
        An existing matplotlib Axes object to draw the plot onto. If not provided, a new Axes object will be created.
    size : int, optional
        Size of the dots in the scatter plot.
    label : str, optional
        Label for the dots in the scatter plot.

    Examples
    --------
    >>> lorentz = LorenzAttractor(nstep=5000)
    >>> lorentz.solve()
    >>> plot_simulation_color_2d(lorentz, x_axis='x', y_axis='y', color='z')
    """
    index = {"x":0, "y":1, "z":2, "t":3}
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot()
    data = lorentz_attractor.simulation

    # Check if x represent a valid color or a variable
    try: 
        assert x_axis in index.keys(), f"color must be one of {index.keys()}"
        x_vals = data[index[x_axis]]
    except:
        assert isinstance(x_axis, (int, float)), f"x_axis must be a numeric if not one of {index.keys()}"
        x_vals = np.array([x_axis]*data[0].shape[0])
    
    # Check if y represent a valid color or a variable
    try: 
        y_vals = data[index[y_axis]]
    except:
        assert isinstance(y_axis, (int, float)), f"y_axis must be a numeric if not one of {index.keys()}"
        y_vals = np.array([y_axis]*data[0].shape[0])
    # Check if color represent a valid color or a variable
    try: 
        assert color in index.keys(), f"color must be one of {index.keys()}"
        colors = cm.terrain(data[index[color]] / max(data[index[color]]))
    except:
        assert is_color_like(color), f"color must be a valid color or one of {index.keys()}"
        colors = color

    ax.scatter(x_vals, y_vals, c=colors, marker = ".", s=size, label=label)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"Lorenz Attractor, method={lorentz_attractor.method.__name__}\n sigma={lorentz_attractor.sigma:.2f}, beta={lorentz_attractor.beta:.2f}, rho={lorentz_attractor.rho:.2f}\ninital: x={lorentz_attractor.x0:.2f}, y={lorentz_attractor.y0:.2f}, z={lorentz_attractor.z0:.2f}", fontsize=10)

def plot_multiple_settings(sigma: list=[10], beta: list=[8/3], rho: list=[16], initial: list=[[0.1,0.1,0.1]], total_time: list=[50], dt: list=[0.01], method: list = ["euler"], ncols: int = 3, dtype: list = [np.float64], ugly_title: bool = False):
    """
    Plots multiple Lorenz Attractor simulations with different settings on the same page.
    
    Each simulation corresponds to a different set of parameters defined by the input lists. If an input list has multiple elements,
    a new Lorenz Attractor simulation is performed for each element. For input lists with only a single element, the same parameter is
    used for all simulations. The plots are arranged in a grid with a specified number of columns.

    Parameters
    ----------
    sigma : list, optional
        List of sigma values to use in Lorenz Attractor simulations.
    beta : list, optional
        List of beta values to use in Lorenz Attractor simulations.
    rho : list, optional
        List of rho values to use in Lorenz Attractor simulations.
    initial : list, optional
        List of initial [x, y, z] coordinates to use in Lorenz Attractor simulations.
    total_time : list, optional
        List of total time durations for Lorenz Attractor simulations.
    dt : list, optional
        List of time step sizes to use in Lorenz Attractor simulations.
    method : list, optional
        List of methods to use for Lorenz Attractor simulations.
    dtype : list, optional
        List of data types to use in Lorenz Attractor simulations.
    ncols : int, optional
        Number of columns for arranging the plots in a grid.
    ugly_title : bool, optional
        If True, titles of the plots will contain all varied simulation parameters.

    Examples
    --------
    >>> plot_multiple_settings(sigma=[10, 15], beta=[8/3, 2.8], rho=[16, 25], initial=[[0.1,0.1,0.1], [0.2,0.2,0.2]], total_time=[50, 75], dt=[0.01, 0.02], method=['euler', 'euler'])
    """
    
    lengths = {
        "sigma":len(sigma),
        "beta":len(beta),
        "rho":len(rho),
        "total_time":len(total_time),
        "dt":len(dt),
        "method":len(method),
        "initial":len(initial),
        "dtype":len(dtype)
    }
    max_len = max(lengths.values())
    iters = {k:v for k, v in lengths.items() if v == max_len}
    assert all([l == 1 or l == max_len for l in list(lengths.values())]), "sigma, beta, rho, total_time, dt, method, and initial must have the same length or length 1"

    _, axs = plt.subplots(
        nrows = (max_len+2)//ncols,
        ncols = ncols,
        figsize= (12, 4 * (max_len+2)//ncols),
        subplot_kw={'projection': '3d'}
    )
    def get_val(values, i):
        return values[min(i, len(values)-1)]
    for i in range(max_len):
        nsteps = int(total_time[min(i, len(total_time)-1)]/dt[min(i, len(dt)-1)])
        sim = LorenzAttractor(
            sigma=get_val(sigma, i), 
            beta=get_val(beta, i),
            rho=get_val(rho, i),
            dt=get_val(dt, i),
            nstep=nsteps,
            method=get_val(method, i),
            x0=get_val(initial, i)[0], 
            y0=get_val(initial, i)[1], 
            z0=get_val(initial, i)[2],
            dtype=get_val(dtype, i)
        )
        sim.solve()
        plot_simulation_color(sim, ax = axs.flatten()[i]);
        if ugly_title:
            axs.flatten()[i].set_title("".join([f"{k}={v}, " for k, v in sim.__dict__.items() if k in iters.keys()]))
        sim.save_simulation("simulations")
    plt.tight_layout()