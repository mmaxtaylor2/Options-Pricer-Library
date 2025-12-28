# core/binomial_tree.py

import math

def binomial_tree_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    steps: int = 200,
    exercise: str = "european",
) -> float:
    """
    Cox–Ross–Rubinstein Binomial Tree Model
    Supports: European & American call/put options
    """

    # Time step
    dt = T / steps

    # Up/down movement factors (CRR model)
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u

    # Risk-neutral probability & discount factor
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    # --- Terminal Stock Prices ---
    prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]

    # --- Terminal Option Payoff ---
    option_type = option_type.lower()
    if option_type == "call":
        values = [max(price - K, 0) for price in prices]
    else:
        values = [max(K - price, 0) for price in prices]

    # --- Backward Induction Through the Tree ---
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            continuation = disc * (p * values[j+1] + (1 - p) * values[j])

            if exercise.lower() == "american":
                intrinsic = (
                    max(prices[j]/u - K, 0) if option_type == "call"
                    else max(K - prices[j]/u, 0)
                )
                values[j] = max(continuation, intrinsic)
            else:
                values[j] = continuation

            prices[j] = prices[j] / u

    return values[0]


# === Quick Self-Test ===
if __name__ == "__main__":
    price = binomial_tree_price(100, 100, 1.0, 0.05, 0.20, "call", 300)
    print("Binomial Price:", price)

