"""
=============================================================================
           MARINE LIFE BIO-ACCUMULATION MODEL (RK4 + PINN)
=============================================================================
*** Although the .py file is uploaded, we do recommend you to use the .ipynb file
to open it directly in the Google Colab. ***

[1] USER GUIDE & INSTRUCTIONS
-----------------------------------------------------------------------------
This script simulates the biological impact of treated water release on marine life.
Unlike the Water Model (passive dispersion), this models 'Bio-accumulation':
the active uptake of radioactive isotopes by fish and their subsequent elimination.

HOW TO OPERATE:
1. Run the script. It is fully automated.
   - Stage 1: Computes the coupled differential equations (Water + Fish).
   - Stage 2: Trains the Neural Network (Log-Space training for stability).
   - Stage 3: Generates a Linear Scale Heatmap (2025-2055).

PREREQUISITES:
   pip install numpy matplotlib torch

Code Running Application:
Google Colab is recommended to run this code. You can download the file and
open it through Google Colab.

[2] HOW TO INTERPRET THE RESULT
-----------------------------------------------------------------------------
The output is a Heatmap showing radioactivity in fish (Bq/kg).
- X-Axis: Distance from Source (km).
- Y-Axis: Time (Years).
- Color:  Red/Orange = Active Uptake zone (~10-20 Bq/kg).
          Blue = Safe/Background levels.
- Key Event: Look for the white dotted line at 2051. You will see a sharp
  drop in color intensity, representing the biological elimination (depuration)
  after the discharge stops.

*** Gemini is used to generate the explanation and the guidelines for clarity.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# PART 1: REALISTIC PHYSICS (Advection-Diffusion)
# ==========================================
# EXPLANATION: This function defines the "Coupled System" of equations.
# We are solving for 4 variables simultaneously at every point in the ocean:
# 1. W_cs: Water Concentration (Cesium)
# 2. F_cs: Fish Concentration (Cesium)
# 3. W_tr: Water Concentration (Tritium)
# 4. F_tr: Fish Concentration (Tritium)
def coupled_derivatives(t, state_vector, params):
    N = params['N']
    u, dx = params['u'], params['dx']

    # DIFFUSION COEFFICIENT (Realistic: 50,000 km^2/yr)
    # Represents the chaotic mixing of the ocean.
    D = 50000.0

    # ISOTOPES & BIOLOGICAL KINETICS
    # lam: Radioactive decay rate (physics).
    # k_up: Uptake rate (biology - how fast fish absorb it).
    # k_elim: Elimination rate (biology - how fast fish pee/sweat it out).

    # Cesium (2011 Legacy): High uptake, slow elimination (bio-accumulates).
    lam_cs, k_up_cs, k_elim_cs = 0.023, 60.0, 1.0
    # Tritium (ALPS): Lower uptake, fast elimination (does not accumulate much).
    lam_tr, k_up_tr, k_elim_tr = 0.056, 30.0, 30.0

    W_cs = state_vector[0:N]
    F_cs = state_vector[N:2*N]
    W_tr = state_vector[2*N:3*N]
    F_tr = state_vector[3*N:4*N]

    # SOURCES (Forcing Functions)
    src_cs = 3000.0 if t <= 0.5 else 0          # 2011 Spike (Fukushima Accident)
    src_tr = 400.0 if (12.0 <= t <= 40.0) else 0 # 2023-2051 Stream (ALPS Release)

    # --- DERIVATIVES ---
    dW_cs = np.zeros(N)
    dW_tr = np.zeros(N)

    # Boundary Conditions (Tank 0 - The Source)
    # Change = Source - Outflow - Decay + Diffusion from neighbor
    dW_cs[0] = src_cs - (u/dx)*W_cs[0] - lam_cs*W_cs[0] + (D/dx**2)*(W_cs[1] - W_cs[0])
    dW_tr[0] = src_tr - (u/dx)*W_tr[0] - lam_tr*W_tr[0] + (D/dx**2)*(W_tr[1] - W_tr[0])

    # Interior Tanks (The Open Ocean)
    for i in range(1, N-1):
        # Diffusion: (Neighbor_Left - 2*Current + Neighbor_Right)
        diff_term = (D/dx**2) * (W_cs[i+1] - 2*W_cs[i] + W_cs[i-1])
        # Advection: Current bringing stuff from the left
        adv_term = (u/dx) * (W_cs[i-1] - W_cs[i])
        dW_cs[i] = adv_term + diff_term - lam_cs*W_cs[i]

        diff_term_tr = (D/dx**2) * (W_tr[i+1] - 2*W_tr[i] + W_tr[i-1])
        adv_term_tr = (u/dx) * (W_tr[i-1] - W_tr[i])
        dW_tr[i] = adv_term_tr + diff_term_tr - lam_tr*W_tr[i]

    # Last Tank (Boundary Condition - Outflow)
    dW_cs[-1] = (u/dx)*(W_cs[-2] - W_cs[-1]) - lam_cs*W_cs[-1]
    dW_tr[-1] = (u/dx)*(W_tr[-2] - W_tr[-1]) - lam_tr*W_tr[-1]

    # Fish Dynamics (The Biological Equation)
    # dFish/dt = (Uptake from Water) - (Elimination) - (Decay)
    dF_cs = (k_up_cs * W_cs) - (k_elim_cs * F_cs) - (lam_cs * F_cs)
    dF_tr = (k_up_tr * W_tr) - (k_elim_tr * F_tr) - (lam_tr * F_tr)

    return np.concatenate((dW_cs, dF_cs, dW_tr, dF_tr))

def generate_realistic_data():
    params = {'N': 60, 'u': 2800.0, 'dx': 170.0}
    dt = 0.05
    T_max = 50.0
    time_vec = np.linspace(0, T_max, int(T_max/dt))

    state = np.zeros(4 * params['N'])
    history = np.zeros((len(time_vec), 4 * params['N']))

    # SOLVER: Runge-Kutta 4 Loop
    for i, t in enumerate(time_vec[:-1]):
        history[i] = state
        k1 = coupled_derivatives(t, state, params)
        k2 = coupled_derivatives(t + 0.5*dt, state + 0.5*dt*k1, params)
        k3 = coupled_derivatives(t + 0.5*dt, state + 0.5*dt*k2, params)
        k4 = coupled_derivatives(t + dt, state + dt*k3, params)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    history[-1] = state
    return time_vec, history, params

print("1. Running Diffusion Physics Model...")
rk4_time, rk4_data, params = generate_realistic_data()

# EXPLANATION: Helper to sum up Cesium + Tritium in fish
def get_total_fish_bq(dist_km, t_year):
    max_dist = params['N'] * params['dx']
    # Small epsilon to avoid log(0) errors later
    min_val = 1e-4
    if t_year <= 0 or dist_km > max_dist: return min_val
    t_idx = int(t_year / 0.05)
    if t_idx >= len(rk4_time): t_idx = -1
    tank_idx = int(dist_km / params['dx'])
    if tank_idx >= params['N']: return min_val

    idx_fish_cs = params['N'] + tank_idx
    idx_fish_tr = 3*params['N'] + tank_idx
    total = rk4_data[t_idx, idx_fish_cs] + rk4_data[t_idx, idx_fish_tr]
    return max(total, min_val)

# ==========================================
# PART 2: TRAIN AI (Log-Train for Stability)
# ==========================================
print("2. Training AI...")
n_samples = 8000
x_train = np.random.uniform(0, 10000, n_samples)
t_train = np.random.uniform(0.1, 50, n_samples)
y_train = np.array([get_total_fish_bq(d, t) for d, t in zip(x_train, t_train)])

# EXPLANATION: Logarithmic Transformation
# Radiation varies from 0.0001 to 100.0. Linear training fails on this range.
# We compress the target data: y_new = log10(y_old).
# This allows the AI to learn both trace amounts and peak amounts equally well.
y_train_log = np.log10(y_train)
x_mean, x_std = x_train.mean(), x_train.std()
t_mean, t_std = t_train.mean(), t_train.std()
y_log_mean, y_log_std = y_train_log.mean(), y_train_log.std()

X_tensor = torch.tensor(np.column_stack(((x_train-x_mean)/x_std, (t_train-t_mean)/t_std)), dtype=torch.float32)
Y_tensor = torch.tensor((y_train_log - y_log_mean)/y_log_std, dtype=torch.float32).view(-1, 1)

model = nn.Sequential(
    nn.Linear(2, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 64), nn.Tanh(),
    nn.Linear(64, 1)
)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for i in range(1500):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(X_tensor), Y_tensor)
    loss.backward()
    optimizer.step()

# ==========================================
# PART 3: LINEAR SCALE MAP
# ==========================================
print("3. Generating Linear Scale Map...")

start_year = 2025
end_year = 2055
sim_start = start_year - 2011
sim_end = end_year - 2011

dist_vals = np.linspace(0, 10000, 300)
time_vals = np.linspace(sim_start, sim_end, 300)
D_grid, T_grid = np.meshgrid(dist_vals, time_vals)

d_flat, t_flat = D_grid.flatten(), T_grid.flatten()
inputs = np.column_stack(((d_flat - x_mean)/x_std, (t_flat - t_mean)/t_std))

with torch.no_grad():
    preds_log_norm = model(torch.tensor(inputs, dtype=torch.float32)).numpy().flatten()

# EXPLANATION: Inverse Transformation
# We must convert the AI's log-prediction back to real numbers (Bq/kg).
# 1. Un-normalize: (Pred * std) + mean
# 2. Un-log: 10^x
preds_raw = 10**((preds_log_norm * y_log_std) + y_log_mean)
Z_grid = preds_raw.reshape(D_grid.shape)

# LINEAR PLOTTING LOGIC
# We clamp the Scale to 20.0 Bq/kg.
# Why? Even if some points hit 25, clamping at 20 makes the
# "10-20 Bq/kg" steady state zone appear visually distinct (Red).
plot_vmax = 20.0
plot_vmin = 0.0

plt.figure(figsize=(10, 8))

# Removed LogNorm. Using standard linear normalization.
plt.imshow(Z_grid, aspect='auto', origin='lower', cmap='turbo',
           extent=[0, 10000, start_year, end_year],
           vmin=plot_vmin, vmax=plot_vmax)

cb = plt.colorbar()
cb.set_label('Total Radioactivity (Bq/kg) [Linear Scale]', rotation=270, labelpad=20)

plt.xlabel('Distance from Japan (km)')
plt.ylabel('Prediction Year')
plt.title(f'Future Prediction (Linear Scale): {start_year}-{end_year}')

plt.axhline(y=2025, color='white', linestyle='-', linewidth=2)
plt.text(200, 2025.5, 'CURRENT STATE', color='white', fontweight='bold', fontsize=8)

plt.axhline(y=2051, color='white', linestyle=':', linewidth=2)
plt.text(200, 2051.5, 'DISCHARGE ENDS', color='white', fontweight='bold', fontsize=8)

plt.tight_layout()
plt.show()

"""
=============================================================================
             PART 4: STATISTICAL VALIDATION (MARINE MODEL)
=============================================================================

[1] SCIENTIFIC OBJECTIVE
-----------------------------------------------------------------------------
This module performs a rigorous statistical audit of the Neural Network.
Because biological systems are non-linear (uptake vs. elimination), we need
to verify that the AI captures these dynamics accurately.

We use two standard metrics:
1. PARITY PLOT (Left Panel):
   - Compares AI Predictions (Y-axis) vs. Physics Truth (X-axis).
   - A perfect model forms a straight diagonal line ($y=x$).
   - Deviations show where the AI struggles (e.g., at very low concentrations).

2. RESIDUAL HISTOGRAM (Right Panel):
   - Shows the distribution of errors (Prediction - Truth).
   - We want a sharp peak at 0. This proves the model is "Unbiased".

[2] CRITICAL STEP: REVERSE-LOG TRANSFORMATION
-----------------------------------------------------------------------------
Recall that we trained the model on Log10 data to handle small numbers.
To evaluate accuracy in the Real World, we must:
   1. Take the AI output (Log Normalized).
   2. Un-normalize it.
   3. Raise it to power of 10 ($10^x$) to get Bq/kg.

[3] PREREQUISITES
-----------------------------------------------------------------------------
- The variable 'model' must be a trained PyTorch neural network. You must run
the first part of the script to train the AI and then initiate this program. If
not, the code will not run because it is based on the previous result. If run
time lost happened, you must rerun the first part and then initiate this part.

Code Running Application:
Google Colab is recommended to run this code. You can download the file and
open it through Google Colab.

*** Gemini is used to generate the explanation and the guidelines for clarity.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import torch

# ==========================================
# 1. GENERATE TEST DATA (Marine Model)
# ==========================================
# EXPLANATION: We generate 2000 NEW random points.
# We include t=50 years to specifically test if the AI learned the
# "Drop-off" phase (Depuration) after the discharge stops in 2051.
n_test = 2000
x_test = np.random.uniform(0, 10000, n_test)
t_test = np.random.uniform(0.1, 50, n_test)

# GET GROUND TRUTH (Physics Engine)
# Using 'get_total_fish_bq' from your Marine Code
y_true = np.array([get_total_fish_bq(d, t) for d, t in zip(x_test, t_test)])

# GET PREDICTION (Neural Network)
# Use the training stats from your Marine Model block
# (x_mean, x_std, t_mean, t_std must match the ones used in Marine training)
# Recalculating approx stats based on the training range (0-10000 km, 0-50 yr)
x_mean_m, x_std_m = 5000.0, 2886.0
t_mean_m, t_std_m = 25.0, 14.4

inputs = np.column_stack(((x_test - x_mean_m)/x_std_m, (t_test - t_mean_m)/t_std_m))

with torch.no_grad():
    preds_norm = model(torch.tensor(inputs, dtype=torch.float32)).numpy().flatten()

# REVERSE LOG TRANSFORMATION
# EXPLANATION: This is the most critical step for this specific model.
# The network predicts 'Normalized Logarithms'. We must reverse the math
# to compare apples-to-apples (Bq/kg vs Bq/kg).
# y_pred = 10 ^ ( (Output * Std) + Mean )
y_true_log = np.log10(y_true)
y_log_mean_m = y_true_log.mean()
y_log_std_m = y_true_log.std()

# Un-normalize: (Pred * Std) + Mean
preds_log = (preds_norm * y_log_std_m) + y_log_mean_m
# Un-log: 10^x
y_pred = 10**preds_log

# ==========================================
# 2. CALCULATE METRICS
# ==========================================
r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
residuals = y_pred - y_true

# ==========================================
# 3. PLOT EVALUATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- PANEL 1: PARITY PLOT ---

ax1.scatter(y_true, y_pred, alpha=0.5, s=15, c='darkorange', edgecolor='k', linewidth=0.3, label='Test Points')
# Plot the 45-degree "Perfect Fit" line
max_axis = max(y_true.max(), y_pred.max())
ax1.plot([0, max_axis], [0, max_axis], 'k--', linewidth=2, label='Ideal ($y=x$)')

ax1.set_title(f'Marine Life Model Accuracy\n$R^2$ Score = {r2:.4f}', fontsize=14, fontweight='bold')
ax1.set_xlabel('Physics Ground Truth (Bq/kg)', fontsize=12)
ax1.set_ylabel('Neural Network Prediction (Bq/kg)', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

# --- PANEL 2: ERROR HISTOGRAM ---
ax2.hist(residuals, bins=40, color='orangered', alpha=0.7, edgecolor='black')
ax2.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero Error')

ax2.set_title(f'Residual Error Distribution\nMSE = {mse:.5f}', fontsize=14, fontweight='bold')
ax2.set_xlabel('Prediction Error (Bq/kg)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()