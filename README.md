## Options Pricer Library

A modular Python library for pricing European and American options using multiple quantitative methods. The project is structured to reflect practical workflows used in derivatives research, quant interviews, and entry-level trading/analyst work.

This library includes three core pricing engines:

Black–Scholes–Merton closed-form valuation
Cox–Ross–Rubinstein binomial tree (European & American)
Monte Carlo simulation using GBM terminal distribution with variance reduction

The models are benchmarked against each other to demonstrate numerical convergence.

## Problem Context

Option pricing models are often taught in isolation, which can obscure the trade-offs between analytical solutions and numerical approximations. This project was built to compare closed-form and simulation-based pricing methods within a single, consistent framework, emphasizing model validation, convergence behavior, and practical implementation considerations.

## Why This Project Matters

This toolkit demonstrates:

Understanding of analytical vs numerical pricing
Ability to validate models through comparison
Practical code organization for a finance/quant environment
Comfort with the underlying math (GBM, risk-neutral valuation, discounting)

## Methods Implemented

Black–Scholes–Merton:	Closed-form, call/put, Greeks, IV solver	Fast, reference benchmark
Binomial Tree (CRR):	European & American, early exercise handling	Lattice-based price evolution
Monte Carlo Simulation:	Terminal GBM distribution, antithetic variates	Stochastic pricing, variance control


## File Structure
Options-Pricer-Library/
├─ core/
│   ├─ black_scholes.py       # Closed-form pricing + Greeks + IV solver
│   ├─ binomial_tree.py       # CRR lattice, American exercise support
│   ├─ monte_carlo.py         # GBM simulation with variance reduction
│   └─ utils.py               # Normal PDF/CDF + d1/d2 helpers
│
├─ examples/
│   └─ compare_models.py      # Runs BSM vs Binomial vs Monte Carlo
│
├─ data/                      # Vol surfaces or market inputs (optional)
├─ tests/                     # Future test hooks (pytest-compatible)
└─ ui/                        # For optional Streamlit interface

## Quick Start

Clone the repository:

git clone https://github.com/mmaxtaylor2/Options-Pricer-Library.git
cd Options-Pricer-Library

Install dependencies:

pip install -r requirements.txt

## Run the Model Comparison

This script validates pricing convergence across models:

python3 -m examples.compare_models

Expected result range:

Black–Scholes (Closed Form):      ≈ 10.45
Binomial Tree (300 steps):        ≈ 10.44
Monte Carlo (50,000 paths):       ≈ 10.40 – 10.55

## Individual Model Usage

Black–Scholes
from core.black_scholes import black_scholes_price
price = black_scholes_price(100, 100, 1, 0.05, 0.20, "call")

Binomial Tree
from core.binomial_tree import binomial_tree_price
price = binomial_tree_price(100, 100, 1, 0.05, 0.20, "call", steps=300)

Monte Carlo
from core.monte_carlo import monte_carlo_european
price = monte_carlo_european(100, 100, 1, 0.05, 0.20, "call", n_paths=50000)
