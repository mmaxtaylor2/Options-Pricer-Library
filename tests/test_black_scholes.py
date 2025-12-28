# tests/test_black_scholes.py

import math

from core.black_scholes import black_scholes_price, implied_volatility


def test_bsm_call_price_atm():
    """
    Basic sanity check:
    S = 100, K = 100, T = 1, r = 5%, sigma = 20%
    Known approximate call price ~ 10.45
    """
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    price = black_scholes_price(S, K, T, r, sigma, option_type="call")
    assert abs(price - 10.4506) < 1e-3


def test_bsm_put_price_otm():
    """
    Another check for a put option.
    """
    S, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.25
    price = black_scholes_price(S, K, T, r, sigma, option_type="put")
    assert abs(price - 12.6616) < 1e-3


def test_put_call_parity():
    """
    Check approximate put-call parity:
    C - P = S - K e^{-rT}
    """
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    call_price = black_scholes_price(S, K, T, r, sigma, option_type="call")
    put_price = black_scholes_price(S, K, T, r, sigma, option_type="put")

    lhs = call_price - put_price
    rhs = S - K * math.exp(-r * T)

    assert abs(lhs - rhs) < 1e-4


def test_implied_volatility_recovery():
    """
    Given a model price at sigma=0.2, implied_volatility should
    recover ~0.2 within tolerance.
    """
    S, K, T, r, true_sigma = 100, 100, 1.0, 0.05, 0.2
    target_price = black_scholes_price(S, K, T, r, true_sigma, option_type="call")

    iv = implied_volatility(
        target_price=target_price,
        S=S,
        K=K,
        T=T,
        r=r,
        option_type="call",
    )

    assert abs(iv - true_sigma) < 1e-4
