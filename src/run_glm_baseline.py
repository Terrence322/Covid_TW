import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

# Settings
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
output_dir = "reports"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
raw_data_dir = "data/raw"
file_case = os.path.join(raw_data_dir, "Age_County_Gender_day_19CoV.csv")

print("Loading Data...")
try:
    df = pd.read_csv(file_case)
    df['發病日'] = pd.to_datetime(df['發病日'])
    
    # Separation
    # Assuming '是否為境外移入' == 1 is Imported
    daily_local = df[df['是否為境外移入'] == 0].groupby('發病日')['確定病例數'].sum()
    daily_imported = df[df['是否為境外移入'] == 1].groupby('發病日')['確定病例數'].sum()
    
    df_model = pd.DataFrame({'local': daily_local, 'imported': daily_imported}).fillna(0)
    
    # Reindex
    all_dates = pd.date_range(start=df['發病日'].min(), end=df['發病日'].max(), freq='D')
    df_model = df_model.reindex(all_dates, fill_value=0)
    df_model.index.name = 'date'
    df_model = df_model.reset_index()
    
    print(f"Data Ready. Shape: {df_model.shape}")
    
    # 2. Features
    df_model['local_lag1'] = df_model['local'].shift(1)
    df_model['local_lag7'] = df_model['local'].shift(7)
    df_model['imported_lag1'] = df_model['imported'].shift(1)
    df_model['imported_lag7'] = df_model['imported'].shift(7)
    
    df_model_clean = df_model.dropna()
    print("Features created.")

    # 3. Modeling
    expr = 'local ~ local_lag1 + local_lag7 + imported_lag1 + imported_lag7'
    
    print("\nTraining Poisson Model...")
    poisson_model = smf.glm(formula=expr, data=df_model_clean, family=sm.families.Poisson()).fit()
    print(poisson_model.summary())
    
    print("\nTraining Negative Binomial Model...")
    nb_model = smf.glm(formula=expr, data=df_model_clean, family=sm.families.NegativeBinomial()).fit()
    print(nb_model.summary())
    
    # 4. Visualization
    df_model_clean['pred_poisson'] = poisson_model.predict(df_model_clean)
    df_model_clean['pred_nb'] = nb_model.predict(df_model_clean)
    
    plt.figure(figsize=(15, 6))
    plt.plot(df_model_clean['date'], df_model_clean['local'], label='Actual Local Cases', color='black', alpha=0.6)
    plt.plot(df_model_clean['date'], df_model_clean['pred_poisson'], label='Poisson Fitted', color='blue', linestyle='--')
    plt.plot(df_model_clean['date'], df_model_clean['pred_nb'], label='Negative Binomial Fitted', color='red', linestyle=':')
    
    plt.title('GLM Baseline: Actual vs Fitted Local Cases')
    plt.xlabel('Date')
    plt.ylabel('Daily Cases')
    plt.legend()
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, "02_glm_baseline_fit.png"))
    print(f"Plot saved to {os.path.join(output_dir, '02_glm_baseline_fit.png')}")
    
    # Zoom
    latest_date = df_model_clean['date'].max()
    zoom_start = latest_date - pd.Timedelta(days=180)
    df_zoom = df_model_clean[df_model_clean['date'] > zoom_start]
    
    plt.figure(figsize=(15, 6))
    plt.plot(df_zoom['date'], df_zoom['local'], label='Actual Local Cases', color='black', alpha=0.6)
    plt.plot(df_zoom['date'], df_zoom['pred_nb'], label='Negative Binomial Fitted', color='red')
    plt.title(f'Zoomed View (Last 180 Days)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "02_glm_baseline_zoom.png"))
    print(f"Zoom plot saved to {os.path.join(output_dir, '02_glm_baseline_zoom.png')}")

except Exception as e:
    print(f"Error: {e}")
