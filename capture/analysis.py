import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Simulated data (replace with your actual time and a_array)
np.random.seed(42)
time = np.linspace(0, 100, 1000)
a_array = np.piecewise(
    time,
    [time < 10, (time >= 10) & (time < 50), (time >= 50)],
    [lambda t: 5 + np.random.normal(0, 1, t.shape),
     lambda t: 10 + np.random.normal(0, 0.2, t.shape),  # stable region
     lambda t: 10 + (t - 50)**1.5 + np.random.normal(0, 1, t.shape)]
)

# --- Parameters
window = 30
std_thresh = 0.3
grad_thresh = 0.05

# --- Rolling std dev and gradient
rolling_std = pd.Series(a_array).rolling(window=window, center=True).std()
grad = np.abs(np.gradient(a_array, time))

# --- Stable where std and gradient are both low
stable_mask = (rolling_std < std_thresh) & (grad < grad_thresh)

# --- Plotting
plt.figure(figsize=(10, 4))
plt.plot(time, a_array, label='Semi-major axis', color='blue')
plt.fill_between(time, a_array.min(), a_array.max(), where=stable_mask,
                 color='green', alpha=0.3, label='Stable region')
plt.xlabel("Time")
plt.ylabel("Semi-major axis (AU)")
plt.title("Stable Region Detection")
plt.legend()
plt.tight_layout()
plt.show()
