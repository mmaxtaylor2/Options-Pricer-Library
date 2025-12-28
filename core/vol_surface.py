# core/vol_surface.py

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .black_scholes import black_scholes_price
from .binomial_tree import binomial_tree_price
from .monte_carlo import monte_carlo_european


class VolSurface:
    """
    Interpolates implied volatilities across strike and maturity.

    Requirements:
    - DataFrame with columns: ["maturity", "strike", "iv"]
    - iv in decimals (e.g., 0.22 for 22%)
    """

    def __init__(self, df: pd.DataFrame):
        if not all(col in df.columns for col in ["maturity", "strike", "iv"]):
            raise ValueError("DataFrame must contain columns: maturity, strike, iv")

        self.df = df.copy()
        self.maturities = np.sort(df["maturity"].unique())
        self.strikes = np.sort(df["strike"].unique())

        # Create grid for interpolation
        grid = df.pivot_table(index="maturity", columns="strike", values="iv")
        self.iv_grid = grid.values

        # Build 2D interpolation function
        self.interpolator = RegularGridInterpolator(
            (self.maturities, self.strikes),
            self.iv_grid,
            bounds_error=False,
            fill_value=None,
        )

    def iv(self, K: float, T: float) -> float:
        """Return interpolated implied volatility."""
        return float(self.interpolator([[T, K]])[0])

    def shock_parallel(self, bump: float):
        """Return a new vol surface shifted by +bump (e.g. +0.02 for +200bps)."""
        bumped_df = self.df.copy()
        bumped_df["iv"] += bump
        return VolSurface(bumped_df)


def price_from_surface(
    S: float,
    K: float,
    T: float,
    r: float,
    vol_surface: VolSurface,
    method: str = "bsm",
    **kwargs,
) -> float:
    """
    Price using implied volatility from the surface.
    """
    sigma = vol_surface.iv(K, T)

    if sigma is None or np.isnan(sigma):
        raise ValueError("Vol surface returned invalid vol: out of interpolation range?")

    method = method.lower()

    if method == "bsm":
        return black_scholes_price(S, K, T, r, sigma, **kwargs)
    elif method == "binomial":
        return binomial_tree_price(S, K, T, r, sigma, **kwargs)
    elif method == "mc":
        return monte_carlo_european(S, K, T, r, sigma, **kwargs)
    else:
        raise ValueError("method must be 'bsm', 'binomial', or 'mc'")
