import matplotlib.pyplot as plt
from matplotlib import cm
from lorenz_attractor.lorenz import LorenzAttractor
import numpy as np

def plot_simulation(lorentz_attractor, ax=None):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*lorentz_attractor.simulation[0:3], color="red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Lorenz Attractor, method={lorentz_attractor.method.__name__}\n sigma={lorentz_attractor.sigma:.2f}, beta={lorentz_attractor.beta:.2f}, rho={lorentz_attractor.rho:.2f}\nx0={lorentz_attractor.x0:.2f}, y0={lorentz_attractor.y0:.2f}, z0={lorentz_attractor.z0:.2f}", fontsize=10)

def plot_simulation_color(lorentz_attractor, ax=None):
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    x, y, z, t = lorentz_attractor.simulation
    colors = cm.terrain(t / max(t))
    ax.scatter(x, y, z, c=colors, marker = ".")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Lorenz Attractor, method={lorentz_attractor.method.__name__}\n sigma={lorentz_attractor.sigma:.2f}, beta={lorentz_attractor.beta:.2f}, rho={lorentz_attractor.rho:.2f}\ninital: x={lorentz_attractor.x0:.2f}, y={lorentz_attractor.y0:.2f}, z={lorentz_attractor.z0:.2f}", fontsize=10)


def plot_multiple_settings(sigma: list=[10], beta: list=[8/3], rho: list=[16], initial: list=[[0.1,0.1,0.1]], total_time: list=[50], dt: list=[0.01], method: list = ["euler"], ncols: int = 3, dtype: list = [np.float64], ugly_title: bool = False):
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
    for i in range(max_len):
        nsteps = int(total_time[min(i, len(total_time)-1)]/dt[min(i, len(dt)-1)])
        sim = LorenzAttractor(
            sigma=sigma[min(i, len(sigma)-1)], 
            beta=beta[min(i, len(beta)-1)],
            rho=rho[min(i, len(rho)-1)],
            dt=dt[min(i, len(dt)-1)],
            nstep=nsteps,
            method=method[min(i, len(method)-1)],
            x0=initial[min(i, len(initial)-1)][0], 
            y0=initial[min(i, len(initial)-1)][1], 
            z0=initial[min(i, len(initial)-1)][2],
            dtype=dtype[min(i, len(dtype)-1)]
        )
        sim.solve()
        plot_simulation_color(sim, ax = axs.flatten()[i]);
        if ugly_title:
            axs.flatten()[i].set_title("".join([f"{k}={v}, " for k, v in sim.__dict__.items() if k in iters.keys()]))
        sim.save_simulation("simulations")
    plt.tight_layout()