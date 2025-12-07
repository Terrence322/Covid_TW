import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import os

sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 資料載入與聚合
if os.path.exists("data"):
    processed_data_dir = "data/processed"
    reports_dir = "reports"
elif os.path.exists("../data"):
    processed_data_dir = "../data/processed"
    reports_dir = "../reports"
else:
    processed_data_dir = r"C:\Users\flash\Downloads\covid\data\processed"
    reports_dir = r"C:\Users\flash\Downloads\covid\reports"

file_case = os.path.join(processed_data_dir, "twcdc_daily_cases_by_location.csv")

print("Loading Data...")
try:
    df = pd.read_csv(file_case, index_col='Date', parse_dates=True)
    
    cities = [str(c).split('_')[0] for c in df.columns]
    df_city = df.groupby(cities, axis=1).sum()
    
    print(f"Cities/Counties: {df_city.columns.tolist()}")
    
    # 2. 設定 Treatment Unit 與 Donor Pool
    target_city = '台北市'
    exclude_cities = [target_city, '新北市', '基隆市', '桃園市']
    donor_pool = [c for c in df_city.columns if c not in exclude_cities]
    
    print(f"Target: {target_city}")
    print(f"Donor Pool ({len(donor_pool)})")
    
    start_date = '2021-01-01'
    intervention_date = '2021-05-15'
    end_date = '2021-08-31'
    
    data_scm = df_city.loc[start_date:end_date].copy()
    
    # 7-day rolling average
    data_scm_smooth = data_scm.rolling(7).mean().fillna(0)
    
    y_target = data_scm_smooth[target_city]
    X_donors = data_scm_smooth[donor_pool]
    
    # Split Pre/Post
    is_pre = data_scm.index < intervention_date
    y_pre = y_target[is_pre]
    X_pre = X_donors[is_pre]
    
    X_post = X_donors[~is_pre]
    y_post = y_target[~is_pre]
    
    # 3. 權重優化
    def loss_func(W, X, y):
        error = y - X @ W
        return np.sum(error**2)
    
    n_donors = X_pre.shape[1]
    W0 = np.ones(n_donors) / n_donors
    bounds = [(0, 1) for _ in range(n_donors)]
    constraints = {'type': 'eq', 'fun': lambda W: np.sum(W) - 1}
    
    res = minimize(loss_func, W0, args=(X_pre, y_pre), 
                   bounds=bounds, constraints=constraints, method='SLSQP')
    
    weights = res.x
    print("Optimal Weights obtained.")
    
    # 4. 合成
    synthetic_target = X_donors @ weights
    
    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(y_target.index, y_target, label=f'Actual {target_city}', color='black', linewidth=2)
    plt.plot(synthetic_target.index, synthetic_target, label=f'Synthetic {target_city}', color='red', linestyle='--')
    plt.axvline(pd.to_datetime(intervention_date), color='grey', linestyle=':', label='Outbreak Start')
    plt.title(f'Regional Synthetic Control: Actual vs Synthetic {target_city}')
    plt.legend()
    
    output_path = os.path.join(reports_dir, '06_scm_plot.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

except Exception as e:
    print(f"Error: {e}")
