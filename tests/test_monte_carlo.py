# tests/test_monte_carlo.py

from core.black_scholes import black_scholes_price
from core.monte_carlo import monte_carlo_european

def test_mc_converges_to_bsm():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    bsm = black_scholes_price(S, K, T, r, sigma, "call")
    mc = monte_carlo_european(S, K, T, r, sigma, "call", n_paths=20000)

    # Should be close with a decent number of paths
    assert abs(bsm - mc) < 1.0
