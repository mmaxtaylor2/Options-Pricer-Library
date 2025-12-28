# tests/test_binomial_tree.py

from core.binomial_tree import binomial_tree_price
from core.black_scholes import black_scholes_price

def test_binomial_converges_to_bsm():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    bsm_price = black_scholes_price(S, K, T, r, sigma, "call")
    tree_price = binomial_tree_price(S, K, T, r, sigma, steps=500, option_type="call")

    # Should be close if steps are high
    assert abs(bsm_price - tree_price) < 0.25  # convergence tolerance
