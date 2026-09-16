"""
Tests for the decay simulation.

One complete test is given as a model. Add the two tests described in the
lab handout (a negative-rate test, and a test against the analytical law).
Run with:  pytest -v
"""

import numpy as np
import pytest
from decay import simulate, simulate_loop


def test_starts_at_N0():
    # at time zero, no atoms have decayed yet
    assert simulate(1000, 0.4)[0] == 1000


# TODO 1: test_rejects_negative_rate
#   Check that calling simulate(...) with a negative lam raises a ValueError.
#   Which pytest tool checks that an error is raised?

def test_negative_rate_raises():
    with pytest.raises(ValueError):
        simulate(1000, -0.1)

# TODO 2: test_matches_law
#   Check that the simulation's AVERAGE over many seeds is close to the
#   physical law  N0 * exp(-lam * t).
#   Which pytest tool compares floating-point values with a tolerance?


def test_average_matches_theory():
    N0 = 1000
    rate = 0.4
    dt = 0.05
    steps = 100
    seeds = range(100)

    results = [
        simulate(N0, rate, dt, steps, seed)[-1]
        for seed in seeds
    ]

    average = sum(results) / len(results)
    t = dt * steps
    expected = N0 * __import__("math").exp(-rate * t)

    assert average == pytest.approx(expected, abs=10)