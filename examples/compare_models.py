# examples/compare_models.py

# =========================================================
#  OPTIONS PRICING MODEL COMPARISON
#  BSM vs BINOMIAL vs MONTE CARLO
# =========================================================

from core.black_scholes import black_scholes_price
from core.binomial_tree import binomial_tree_price
from core.monte_carlo import monte_carlo_european


# ---------------------------------------------------------
# Shared Input Parameters
# ---------------------------------------------------------
S = 100       # Spot price
K = 100       # Strike
T = 1.0       # Time to maturity (years)
r = 0.05      # Risk-free interest rate
sigma = 0.20  # Volatility
option_type = "call"

print("\n>>> Running model comparison...")
print("Step 1: Loaded inputs successfully.")


# ---------------------------------------------------------
# Model Runs + Debug Checkpoints
# ---------------------------------------------------------
try:
    bsm_price = black_scholes_price(S, K, T, r, sigma, option_type)
    print("Step 2: BSM model completed.")
except Exception as e:
    print("BSM ERROR:", e)
    raise SystemExit()

try:
    binomial_price = binomial_tree_price(S, K, T, r, sigma, option_type, steps=300)
    print("Step 3: Binomial model completed.")
except Exception as e:
    print("BINOMIAL ERROR:", e)
    raise SystemExit()

try:
    mc_price = monte_carlo_european(S, K, T, r, sigma, option_type, n_paths=10000)
    print("Step 4: Monte Carlo model completed.")
except Exception as e:
    print("MONTE CARLO ERROR:", e)
    raise SystemExit()


# ---------------------------------------------------------
# Display Comparison Table
# ---------------------------------------------------------
print("\n=================== MODEL COMPARISON ===================")
print(f"Black–Scholes (Closed Form):      {bsm_price:,.4f}")
print(f"Binomial Tree (300 steps):        {binomial_price:,.4f}")
print(f"Monte Carlo (10,000 paths):       {mc_price:,.4f}")
print("========================================================")

diff_mc = abs(mc_price - bsm_price)
print(f"MC vs BSM difference: {diff_mc:,.4f}\n")

print(">>> Comparison complete.\n")

