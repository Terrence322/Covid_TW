import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# Paths
# Assuming running from src/ or project root. Let's make it robust to being run from project root.
if os.path.exists("data"):
    # Running from root
    processed_data_dir = "data/processed"
    reports_dir = "reports"
elif os.path.exists("../data"):
    # Running from src
    processed_data_dir = "../data/processed"
    reports_dir = "../reports"
else:
    # Fallback absolute
    processed_data_dir = r"C:\Users\flash\Downloads\covid\data\processed"
    reports_dir = r"C:\Users\flash\Downloads\covid\reports"

file_case = os.path.join(processed_data_dir, "twcdc_daily_cases_by_location.csv")

print(f"Loading data from: {file_case}")

try:
    df = pd.read_csv(file_case, index_col='Date', parse_dates=True)
    total_cases = df.sum(axis=1)
    cumulative_cases = total_cases.cumsum()
    
    model_data = pd.DataFrame({
        'Date': total_cases.index,
        'Daily': total_cases.values,
        'Cumulative': cumulative_cases.values,
        'Days': np.arange(len(total_cases))
    })
    
    # Fitting Period: 2021-05-01 to 2021-08-31
    start_date = '2021-05-01'
    end_date = '2021-08-31'
    
    mask = (model_data['Date'] >= start_date) & (model_data['Date'] <= end_date)
    fit_data = model_data.loc[mask].reset_index(drop=True)
    fit_data['Days'] = np.arange(len(fit_data))
    
    print(f"Fitting Period: {start_date} to {end_date}, Days: {len(fit_data)}")
    
    # SEIR Model
    def seir_model(y, t, N, beta, sigma, gamma):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt
    
    N = 23500000
    sigma = 1.0 / 5.2
    gamma = 1.0 / 10.0
    
    def fit_seir(params, data):
        beta, I0 = params
        E0 = I0
        R0_init = 0
        S0 = N - E0 - I0 - R0_init
        y0 = S0, E0, I0, R0_init
        
        t = data['Days'].values
        ret = odeint(seir_model, y0, t, args=(N, beta, sigma, gamma))
        S, E, I, R = ret.T
        
        # Fit to Cumulative Cases (approximated as N - S)
        pred_cumulative = N - S
        
        # Loss function: MSE
        # Use offset to align start
        start_val = data['Cumulative'].iloc[0]
        # Align pred_cumulative to start at roughly the same point or just fit increment?
        # The prompt code subtracted start_cases from data. Let's do that.
        data_shifted = data['Cumulative'].values - start_val
        pred_shifted = pred_cumulative - pred_cumulative[0] # simple alignment
        
        # Actually, let's follow the notebook logic: 
        # fit_data_shifted['Cumulative'] = fit_data['Cumulative'] - start_cases
        # and implicit expectation that pred starts near 0.
        
        loss = np.mean((pred_cumulative - (data['Cumulative'].values - start_val))**2) # This assumes pred starts at 0, which it does (N - (N-I0-E0) = small)
        return loss

    initial_guess = [0.2, 100]
    bounds = [(0.0, 1.0), (1, 10000)]
    
    # Shift data for fitting as per notebook logic intent
    fit_data_shifted = fit_data.copy()
    start_cases = fit_data['Cumulative'].iloc[0]
    
    # Optimization
    # Note: The loss function defined above essentially fits the *shape* and *growth* starting from ~0 
    # to match the *growth* of real data starting from 0 (after shift).
    
    res = minimize(fit_seir, initial_guess, args=(fit_data,), bounds=bounds, method='L-BFGS-B')
    
    beta_opt, I0_opt = res.x
    R0_reproduction = beta_opt / gamma
    
    print(f"Optimal Beta: {beta_opt:.4f}")
    print(f"Optimal I0: {I0_opt:.2f}")
    print(f"Basic Reproduction Number (R0): {R0_reproduction:.2f}")
    
    # Simulation
    t_sim = np.linspace(0, 150, 150)
    
    # Fitted
    y0_fit = (N - I0_opt*2, I0_opt, I0_opt, 0)
    ret_fit = odeint(seir_model, y0_fit, t_sim, args=(N, beta_opt, sigma, gamma))
    S_fit, E_fit, I_fit, R_fit = ret_fit.T
    cum_fit = N - S_fit + start_cases
    
    # Counterfactual (Beta + 20%)
    beta_cf = beta_opt * 1.2
    ret_cf = odeint(seir_model, y0_fit, t_sim, args=(N, beta_cf, sigma, gamma))
    S_cf, E_cf, I_cf, R_cf = ret_cf.T
    cum_cf = N - S_cf + start_cases
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(fit_data['Days'], fit_data['Cumulative'], label='Actual Data', color='black', s=10)
    plt.plot(t_sim, cum_fit, label=f'Fitted SEIR (R0={R0_reproduction:.2f})', color='blue')
    plt.plot(t_sim, cum_cf, label=f'Counterfactual: +20% Beta (R0={beta_cf/gamma:.2f})', color='red', linestyle='--')
    
    plt.title('SEIR Model Calibration & Counterfactual Simulation (May-Aug 2021)')
    plt.xlabel('Days since May 1, 2021')
    plt.ylabel('Cumulative Cases')
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(reports_dir, '05_seir_simulation.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

except Exception as e:
    print(f"An error occurred: {e}")
