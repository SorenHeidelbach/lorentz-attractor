import pytest
from lorenz_attractor.lorenz_attractor import LorenzAttractor
from lorenz_attractor.data import load_lorenz_simulation
from scipy.integrate import odeint

@pytest.fixture
def attractor():
    return LorenzAttractor()

def test_initialization(attractor):
    assert attractor.x0 == 0.1
    assert attractor.y0 == 0.1
    assert attractor.z0 == 0.1
    assert attractor.nstep == 5000
    assert attractor.dt == 0.01
    assert attractor.sigma == 10
    assert attractor.beta == 8 / 3
    assert attractor.rho == 6
    assert attractor.simulation is None
    assert attractor.method in [attractor.euler, attractor.rk4]

def test_define_method(attractor):
    attractor.define_method("euler")
    assert attractor.method == attractor.euler

    attractor.define_method("rk4")
    assert attractor.method == attractor.rk4

def test_step_x(attractor):
    x = attractor.step_x(0.1, 0.2, 0.3)
    assert x == pytest.approx(0.1 + 0.01 * 10 * (0.2 - 0.1))

def test_step_y(attractor):
    y = attractor.step_y(0.1, 0.2, 0.3)
    assert y == pytest.approx(0.2 + 0.01 * (0.1 * (6 - 0.3) - 0.2))

def test_step_z(attractor):
    z = attractor.step_z(0.1, 0.2, 0.3)
    assert z == pytest.approx(0.3 + 0.01 * (0.1 * 0.2 - (8/3) * 0.3))

def test_euler(attractor):
    x, y, z = attractor.euler(0.1, 0.2, 0.3)
    assert (x, y, z) == pytest.approx((0.11000000000000001, 0.20370000000000002, 0.2922))
    # Test the expected values of x, y, and z after one Euler step

def test_rk4(attractor):
    x, y, z = attractor.rk4(0.1, 0.2, 0.3)
    assert (x, y, z) == pytest.approx((0.10110598992306367, 0.20204741736944853, 0.302936288053705))
    # Test the expected values of x, y, and z after one RK4 step

def test_solve(attractor):
    attractor.solve()
    assert attractor.simulation is not None
    x, y, z, t = attractor.simulation
    assert len(x) == attractor.nstep
    assert len(y) == attractor.nstep
    assert len(z) == attractor.nstep
    assert len(t) == attractor.nstep
    # Test the values of x, y, and z for the simulation



import os
import unittest
import numpy as np



def test_load_simulation():
    # Test loading of a correct simulation file
    loaded_data = load_lorenz_simulation("test/data/test_sim.npz")
    sim = LorenzAttractor().solve()
    np.testing.assert_array_equal(np.array(loaded_data)["sim"], np.array(sim.simulation))


def test_load_simulation_wrong_filename(self):
    # Test assertion when providing a wrong filename
    with self.assertRaises(AssertionError):
        load_lorenz_simulation("test/data/no_file.npz")

def test_load_simulation_wrong_extension(self):
    # Test assertion when providing a filename with a wrong extension
    with self.assertRaises(AssertionError):
        load_lorenz_simulation("wrong_extension.txt")

def test_load_simulation_wrong_data(self):
    # Test assertion when the npz file does not contain the right data
    wrong_filename = "wrong_data.npz"
    data = np.array([1, 2, 3, 4, 5])
    np.savez(wrong_filename, wrong_key=data)

    with self.assertRaises(AssertionError):
        load_lorenz_simulation(wrong_filename)
    
    os.remove(wrong_filename)
