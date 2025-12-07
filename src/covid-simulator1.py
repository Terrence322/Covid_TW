import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import gc
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 第一部分：模型訓練函數
# ==========================================
def train_covid_model_optimized(file_path):
    print("1. [訓練階段] 正在讀取資料...")
    use_cols = ['發病日', '縣市', '鄉鎮', '確定病例數']
    try:
        df = pd.read_csv(file_path, usecols=use_cols)
    except ValueError:
        df = pd.read_csv(file_path)
        df = df[use_cols]
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")
        return None, None

    # 資料清理
    df['確定病例數'] = pd.to_numeric(df['確定病例數'], errors='coerce').fillna(0).astype(int)
    df = df[df['確定病例數'] > 0].copy()
    
    # 建立地點 ID
    df['Location'] = df['縣市'].astype(str) + "_" + df['鄉鎮'].astype(str)
    le = LabelEncoder()
    df['Location_Code'] = le.fit_transform(df['Location'])
    
    # 時間處理
    df['發病日'] = pd.to_datetime(df['發病日'])
    df['DayOfYear'] = df['發病日'].dt.dayofyear
    
    # 精簡資料
    df_slim = df[['Location_Code', 'DayOfYear', '確定病例數']].copy()
    del df
    gc.collect()
    
    # 展開資料
    print("2. [訓練階段] 正在展開與取樣資料...")
    try:
        # 取樣 30% 的資料來訓練以節省時間與記憶體，同時保持隨機性
        df_temp = df_slim.loc[df_slim.index.repeat(df_slim['確定病例數'])]
        df_expanded = df_temp.sample(frac=0.3, random_state=42).reset_index(drop=True)
        del df_temp
    except MemoryError:
        print("記憶體不足，請降低取樣比例。")
        return None, None
        
    del df_slim
    gc.collect()

    # 特徵工程
    df_expanded = df_expanded.sort_values('DayOfYear').reset_index(drop=True)
    df_expanded['Prev_Location_Code'] = df_expanded['Location_Code'].shift(1)
    df_expanded['Prev_DayOfYear'] = df_expanded['DayOfYear'].shift(1)
    
    df_model = df_expanded.dropna().copy()
    df_model['Prev_Location_Code'] = df_model['Prev_Location_Code'].astype(int)
    df_model['Prev_DayOfYear'] = df_model['Prev_DayOfYear'].astype(int)

    X = df_model[['Prev_Location_Code', 'Prev_DayOfYear', 'DayOfYear']]
    y = df_model['Location_Code']
    
    del df_expanded
    gc.collect()

    print("3. [訓練階段] 正在訓練模型...")
    # 使用較多的樹 (n_estimators) 來捕捉更細膩的機率分佈
    clf = RandomForestClassifier(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    
    print("模型訓練完成！")
    return clf, le

# ==========================================
# 第二部分：模擬器引擎函數 (核心改進)
# ==========================================
def run_pandemic_simulation(model, le, start_locs, start_date_str, num_cases_to_simulate):
    """
    start_locs: 支援多個起始點的列表，例如 ["台北市_內湖區", "高雄市_小港區"]
    """
    print(f"\n4. [模擬階段] 開始生成 {num_cases_to_simulate} 筆模擬資料 (使用活躍池與機率擴散)...")
    print(f"   起始爆發點: {start_locs}")
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    start_day_of_year = start_date.timetuple().tm_yday
    all_classes = model.classes_ # 這些是 Location Code (int)
    
    # 初始化活躍傳染源池 (Active Pool)
    # 每個元素包含: {Location Code, Current Date, DayOfYear}
    active_pool = []
    
    for loc in start_locs:
        if loc in le.classes_:
            loc_code = le.transform([loc])[0]
            active_pool.append({
                "Location_Code": loc_code,
                "Date": start_date,
                "DayOfYear": start_day_of_year
            })
        else:
            print(f"警告: 起始點 {loc} 不在訓練資料中，隨機選擇替代。")
            random_code = np.random.choice(all_classes)
            active_pool.append({
                "Location_Code": random_code,
                "Date": start_date,
                "DayOfYear": start_day_of_year
            })

    simulation_results = []
    
    # 用於進度顯示
    step = max(1, num_cases_to_simulate // 10)

    for i in range(num_cases_to_simulate):
        # 1. 從活躍池中隨機選擇一個傳染源 (Spreader)
        # 為了模擬時間推進，我們傾向選擇較新的案例，但也保留舊案例復燃的可能性
        # 這裡簡單起見，從最近的 50 個活躍案例中隨機選
        pool_size = len(active_pool)
        recent_limit = min(pool_size, 100)
        spreader_idx = random.randint(pool_size - recent_limit, pool_size - 1)
        spreader = active_pool[spreader_idx]
        
        current_loc_code = spreader["Location_Code"]
        current_doy = spreader["DayOfYear"]
        current_date_obj = spreader["Date"]
        
        # 2. 決定時間間隔 (0~2天)
        days_delta = np.random.choice([0, 1, 1, 2], p=[0.2, 0.4, 0.3, 0.1])
        next_date = current_date_obj + timedelta(days=int(days_delta))
        next_doy = next_date.timetuple().tm_yday
        
        # 3. [核心改進] 使用 predict_proba 進行機率性預測
        input_data = [[current_loc_code, current_doy, next_doy]]
        
        try:
            # 取得所有地點的機率分佈
            probs = model.predict_proba(input_data)[0]
            
            # 加上微小的平滑值 (Smoothing)，防止某些機率為 0 導致無法擴散
            # 這相當於極低機率的「長距離跳躍」或「未知傳播」，替代原本的 Chaos Rate
            probs = probs + 1e-6 
            probs = probs / probs.sum() # 重新歸一化
            
            # 依機率抽樣下一個地點代碼
            next_loc_code = np.random.choice(all_classes, p=probs)
            
        except Exception as e:
            # 若出錯，退回到隨機
            next_loc_code = np.random.choice(all_classes)
        
        # 轉回地名
        next_loc_name = le.inverse_transform([next_loc_code])[0]
        
        # 4. 記錄結果
        result_entry = {
            "Case_ID": i + 1,
            "Date": next_date.strftime("%Y-%m-%d"),
            "Location": next_loc_name,
            "DayOfYear": next_doy,
            "Source_Location": le.inverse_transform([current_loc_code])[0] # 追蹤來源
        }
        simulation_results.append(result_entry)
        
        # 5. 將新病例加入活躍池
        active_pool.append({
            "Location_Code": next_loc_code,
            "Date": next_date,
            "DayOfYear": next_doy
        })
        
        # 6. 管理活躍池大小 (避免無限增長變慢，且模擬病毒世代更迭)
        if len(active_pool) > 500:
            active_pool.pop(0) # 移除最舊的
            
        if (i + 1) % step == 0:
            print(f"-> 已生成 {i + 1} 筆... 最新案例位置: {next_loc_name}")

    return pd.DataFrame(simulation_results)

# ==========================================
# 主程式執行區
# ==========================================
if __name__ == "__main__":
    filename = 'Age_County_Gender_day_19CoV.csv' 
    
    # 1. 訓練
    model, le = train_covid_model_optimized(filename)
    
    if model is not None:
        # 2. 設定多個爆發點 (模擬機場或主要城市)
        # 您可以自由新增更多地點
        start_locations = [
            "台北市_松山區",   # 松山機場周邊
            "桃園市_大園區",   # 桃園機場
            "高雄市_小港區"    # 小港機場
        ]
        
        # 3. 模擬
        simulated_df = run_pandemic_simulation(
            model=model, 
            le=le, 
            start_locs=start_locations,
            start_date_str="2024-01-01",
            num_cases_to_simulate=10000 # 先跑一萬筆測試
        )
        
        # 4. 存檔
        output_file = "simulated_outbreak.csv"
        # 儲存需要的欄位
        simulated_df = simulated_df[['Case_ID', 'Date', 'Location', 'DayOfYear']]
        simulated_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n成功！模擬結果已儲存至: {output_file}")
        print(simulated_df.head())
