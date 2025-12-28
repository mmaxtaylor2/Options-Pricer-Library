# core/black_scholes.py

"""
Black–Scholes–Merton European option pricer, Greeks, and implied volatility.

Assumptions:
- No dividends
- European-style exercise
- Constant risk-free rate and volatility
"""

from __future__ import annotations

from typing import Dict

from .utils import normal_cdf, normal_pdf, d1, d2


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Price a European call or put using the Black–Scholes–Merton model.

    Parameters
    ----------
    S : float
        Spot price of the underlying.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualized, decimal).
    option_type : {"call", "put"}
        Type of option.

    Returns
    -------
    float
        Theoretical option price.
    """
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    option_type = option_type.lower()
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)

    from math import exp

    if option_type == "call":
        return S * normal_cdf(d_1) - K * exp(-r * T) * normal_cdf(d_2)
    elif option_type == "put":
        return K * exp(-r * T) * normal_cdf(-d_2) - S * normal_cdf(-d_1)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> Dict[str, float]:
    """
    Compute the main Greeks for a European option under Black–Scholes.

    Returns a dict with:
    - delta
    - gamma
    - vega  (per 1 unit of volatility, i.e. sigma + 1.0)
    - theta (per year)
    - rho   (per 1.0 change in rate)

    Note: No dividends, European exercise.
    """
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    option_type = option_type.lower()
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)

    from math import exp

    pdf_d1 = normal_pdf(d_1)
    cdf_d1 = normal_cdf(d_1)
    cdf_minus_d1 = normal_cdf(-d_1)
    cdf_d2 = normal_cdf(d_2)
    cdf_minus_d2 = normal_cdf(-d_2)

    # Common terms
    discount = exp(-r * T)

    # Greeks that are the same for call and put except sign changes
    gamma = pdf_d1 / (S * sigma * (T ** 0.5))
    vega = S * pdf_d1 * (T ** 0.5)  # per 1.0 change in volatility

    if option_type == "call":
        delta = cdf_d1
        theta = (
            - (S * pdf_d1 * sigma) / (2.0 * (T ** 0.5))
            - r * K * discount * cdf_d2
        )
        rho = K * T * discount * cdf_d2
    elif option_type == "put":
        delta = cdf_d1 - 1.0
        theta = (
            - (S * pdf_d1 * sigma) / (2.0 * (T ** 0.5))
            + r * K * discount * cdf_minus_d2
        )
        rho = -K * T * discount * cdf_minus_d2
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def implied_volatility(
    target_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
    lower: float = 1e-6,
    upper: float = 5.0,
) -> float:
    """
    Solve for the implied volatility given a market price using bisection.

    Parameters
    ----------
    target_price : float
        Observed market price of the option.
    S, K, T, r, option_type
        Same as in black_scholes_price.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    lower, upper : float
        Initial bounds for volatility search.

    Returns
    -------
    float
        Implied volatility.

    Raises
    ------
    ValueError
        If a solution cannot be found within the bounds.
    """
    if target_price <= 0:
        raise ValueError("target_price must be positive.")

    option_type = option_type.lower()

    def f(sigma_val: float) -> float:
        return black_scholes_price(S, K, T, r, sigma_val, option_type) - target_price

    # Check if the initial bracket actually contains a root
    f_lower = f(lower)
    f_upper = f(upper)

    if f_lower * f_upper > 0:
        raise ValueError(
            "Implied vol not bracketed: try expanding lower/upper bounds."
        )

    a, b = lower, upper

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        f_mid = f(mid)

        if abs(f_mid) < tol:
            return mid

        # Bisection: choose the subinterval that contains the root
        if f_lower * f_mid < 0:
            b = mid
            f_upper = f_mid
        else:
            a = mid
            f_lower = f_mid

    # If we exit the loop, we didn't converge
    raise ValueError("Implied volatility did not converge within max_iter.")
