# core/utils.py

import math


def normal_pdf(x: float) -> float:
    """
    Standard normal probability density function.
    """
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def normal_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.
    Uses math.erf for numerical stability.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black–Scholes d1 term (no dividends).
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black–Scholes d2 term (no dividends).
    """
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)
