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
# 第一部分：模型訓練函數 (之前優化過的版本)
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

    # [新增] 過濾日期範圍：2020-01-11 至 2023-12-26
    start_filter = pd.Timestamp("2020-01-11")
    end_filter = pd.Timestamp("2023-12-26")
    df = df[(df['發病日'] >= start_filter) & (df['發病日'] <= end_filter)]
    print(f"資料過濾後範圍: {df['發病日'].min().date()} 至 {df['發病日'].max().date()}，剩餘筆數: {len(df)}")

    df['DayOfYear'] = df['發病日'].dt.dayofyear
    
    # 精簡資料
    df_slim = df[['Location_Code', 'DayOfYear', '確定病例數']].copy()
    del df
    gc.collect()
    
    # 展開資料 (使用採樣以防記憶體不足)
    print("2. [訓練階段] 正在展開與取樣資料...")
    try:
        # 如果記憶體足夠，使用全部；若不足，可取消下一行的註解改用 sample
        # df_expanded = df_slim.loc[df_slim.index.repeat(df_slim['確定病例數'])].reset_index(drop=True)
        
        # 為了保險起見，我們預設只取 20% 的資料來訓練，速度快且不易當機
        df_temp = df_slim.loc[df_slim.index.repeat(df_slim['確定病例數'])]
        df_expanded = df_temp.sample(frac=0.1, random_state=42).reset_index(drop=True)
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
    # 輕量化參數
    clf = RandomForestClassifier(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    
    print("模型訓練完成！")
    return clf, le

# ==========================================
# 第二部分：模擬器引擎函數 (升級為多點活躍池模式)
# ==========================================
def run_pandemic_simulation(model, le, start_locs, start_date_str, num_cases_to_simulate, chaos_level=0.01):
    """
    升級版模擬核心：
    1. 支援多點爆發 (start_locs 可以是列表)
    2. 使用「活躍傳染源池 (Active Pool)」取代單線程傳播，模擬真實群體感染
    3. 結合機率性預測 (predict_proba) 解決確定性陷阱
    """
    if isinstance(start_locs, str):
        start_locs = [start_locs]
        
    print(f"\n4. [模擬階段] 開始生成 {num_cases_to_simulate} 筆模擬資料...")
    print(f"   起始爆發點: {start_locs}")
    print(f"   混亂指數 (Chaos): {chaos_level}")
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    start_doy = start_date.timetuple().tm_yday
    all_locations = le.classes_
    
    # 初始化活躍傳染源池 (Active Pool)
    # 結構: [{'Location_Code': int, 'Date': datetime, 'DayOfYear': int}, ...]
    active_pool = []
    
    for loc in start_locs:
        if loc in le.classes_:
            loc_code = le.transform([loc])[0]
        else:
            print(f"警示: 起始點 {loc} 未知，隨機選擇替代。")
            loc_code = np.random.choice(range(len(all_locations)))
            
        active_pool.append({
            "Location_Code": loc_code,
            "Date": start_date,
            "DayOfYear": start_doy
        })
    
    simulation_results = []
    step_size = max(1, num_cases_to_simulate // 10)
    
    for i in range(num_cases_to_simulate):
        # 1. [核心機制] 從活躍池中隨機選擇傳播者 (Spreader)
        # 這打破了單線程的「A傳B傳C」限制，變成「A傳給B, A也傳給C, B傳給D...」
        spreader = random.choice(active_pool)
        
        current_loc_code = spreader["Location_Code"]
        current_doy = spreader["DayOfYear"]
        current_date = spreader["Date"]
        
        # 2. 決定時間推移 (0~2天)
        days_delta = np.random.choice([0, 1, 1, 2], p=[0.2, 0.4, 0.3, 0.1])
        next_date = current_date + timedelta(days=int(days_delta))
        next_doy = next_date.timetuple().tm_yday
        
        next_loc_code = None
        
        # 3. 預測下一個地點 (結合 Chaos 與 Model)
        if random.random() < chaos_level:
            # 隨機擴散 (模擬旅行、遠距離移動)
            next_loc_code = np.random.choice(range(len(all_locations)))
        else:
            # 模型規律擴散
            input_data = [[current_loc_code, current_doy, next_doy]]
            try:
                # 使用機率分佈抽樣 (解決確定性陷阱)
                probs = model.predict_proba(input_data)[0]
                probs = probs + 1e-7 # 平滑處理
                probs = probs / probs.sum()
                next_loc_code = np.random.choice(model.classes_, p=probs)
            except:
                next_loc_code = np.random.choice(range(len(all_locations)))

        next_loc_name = le.inverse_transform([next_loc_code])[0]
        source_loc_name = le.inverse_transform([current_loc_code])[0]
        
        # 4. 記錄結果
        simulation_results.append({
            "Case_ID": i + 1,
            "Date": next_date.strftime("%Y-%m-%d"),
            "Location": next_loc_name,
            "DayOfYear": next_doy,
            "Source_Location": source_loc_name,
            "Type": "Simulated"
        })
        
        # 5. 將新病例加入活躍池
        active_pool.append({
            "Location_Code": next_loc_code,
            "Date": next_date,
            "DayOfYear": next_doy
        })
        
        # 6. 管理活躍池 (移除舊傳播者，模擬病毒世代交替)
        # 維持池子大小在 50~200 之間，模擬同時具有傳染力的人數
        if len(active_pool) > 200:
            active_pool.pop(0)

        if (i + 1) % step_size == 0:
            print(f"-> 已生成 {i + 1} 筆... 最新: {next_loc_name} (源自: {source_loc_name})")

    return pd.DataFrame(simulation_results)

# ==========================================
# 主程式執行區
# ==========================================
if __name__ == "__main__":
    # 檔名請確認是否正確
    filename = 'Age_County_Gender_day_19CoV.csv' 
    
    # 1. 先訓練 (產生 model 和 le)
    model, le = train_covid_model_optimized(filename)
    
    if model is not None:
        # 2. 再模擬 (使用剛產生的 model)
        # 設定多個起始爆發點 (模擬真實情況)
        start_locations = ["台北市_內湖區", "新北市_板橋區", "桃園市_中壢區"]

        simulated_df = run_pandemic_simulation(
            model=model, 
            le=le, 
            start_locs=start_locations,   # 改為傳入列表
            start_date_str="2024-01-01", 
            num_cases_to_simulate=200000, 
            chaos_level=0.05              # 降低混亂度，依賴模型本身的機率擴散
        )
        
        # 3. 存檔
        output_file = "simulated_outbreak.csv"
        simulated_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n成功！模擬結果已儲存至: {output_file}")
        print(simulated_df.head())