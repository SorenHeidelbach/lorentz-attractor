import numpy as np
import pytest
from lorenz_attractor.lorenz import LorenzAttractor
from scipy.integrate import odeint
from hypothesis import given, strategies as st

def lorenz_scipy(w, t, p):
    x, y, z = w
    sigma, beta, rho = p
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

@given(st.floats(0.01, 0.1), st.floats(0.01, 0.1), st.floats(0.01, 0.1), st.integers(2, 200), st.floats(0.00001, 0.0001), st.floats(5, 15), st.floats(2, 10), st.floats(6, 26))
def test_step_methods(x, y, z, nstep, dt, sigma, beta, rho):
    lorenz = LorenzAttractor(x0=x, y0=y, z0=z, nstep=nstep, dt=dt, sigma=sigma, beta=beta, rho=rho)
    t = np.linspace(0, (nstep-1)*dt, nstep)
    scipy_sol = odeint(lorenz_scipy, (x, y, z), t, args=((sigma, beta, rho),))

    lorenz.solve()

    assert np.allclose(scipy_sol[:, 0], lorenz.simulation[0], atol=1e-4)
    assert np.allclose(scipy_sol[:, 1], lorenz.simulation[1], atol=1e-4)
    assert np.allclose(scipy_sol[:, 2], lorenz.simulation[2], atol=1e-4)

