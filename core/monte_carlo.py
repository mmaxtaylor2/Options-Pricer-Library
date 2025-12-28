# core/monte_carlo.py

import numpy as np
import math

"""
Monte Carlo Option Pricing (European)
-------------------------------------
This implementation uses the *closed form terminal GBM formula*:

    S_T = S0 * exp((r - 0.5σ²)T + σ√T * Z)

This eliminates time-step bias and ensures convergence
towards the Black–Scholes price as n_paths → large.
"""


def monte_carlo_european(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_paths: int = 50000,
    antithetic: bool = True
) -> float:

    # Generate standard normal draws
    Z = np.random.randn(n_paths)

    # Antithetic variance reduction (mirrors distribution)
    if antithetic:
        Z = np.concatenate([Z, -Z])

    # Terminal stock price using GBM closed-form
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)

    # Payoff
    if option_type.lower() == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    # Discounted expected value
    return math.exp(-r * T) * payoff.mean()


# =============================
# Self-test (optional)
# =============================
if __name__ == "__main__":
    price = monte_carlo_european(100, 100, 1, 0.05, 0.20, "call", 50000)
    print("Monte Carlo test price:", price)

