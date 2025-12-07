# 開發日誌 Development Log

## 2025-12-06 初步分析與修復計畫

### 目前問題 (Current Issues)
1. **空間過度擬合（Spatial Overfitting/Clustering）：** 隨機森林模型的 `predict()` 方法預設回傳機率最高的類別。由於傳染病數據具有高度的空間連續性，導致模型幾乎總是預測下一個確診者位於與上一個確診者完全相同的地點。
2. **不自然的混亂機制（Unnatural Chaos）：** 目前的修復方式依賴 `chaos_level`，這會導致確診點在地圖上隨機瞬間移動，忽略了地理距離因素。
3. **單一傳播鏈限制（Single Chain Limitation）：** 目前的模擬僅遵循單一鏈條 (`Case A -> Case B -> Case C`)。這使得無法模擬多個同時發生的爆發點（例如多個機場同時有案例），也忽略了一個人可能傳染給多人的事實。

### 預計改進項目 (Planned Improvements)
1. **機率性抽樣（Probabilistic Sampling）：** 將 `model.predict()` 替換為 `model.predict_proba()`。利用這些機率進行加權隨機抽樣來決定下一個地點。這能讓病毒自然擴散到鄰近的高機率區域，而無需依賴人為的「混亂率」。
2. **活躍病例池（Active Case Pool）：** 實作一個 `active_cases` 清單（近期感染者）。在每個模擬步驟中，從此池中隨機選擇一位「傳播者」來預測下一個案例。此架構本質上支援 **多點爆發**（只需在池中初始化位於不同城市的案例即可）。

### 技術變更 (Technical Changes)
- **檔案：** `covid/covid-simulator1.py`
- **函數：** `run_pandemic_simulation` 與 `simulate_next_location`
- **動作：** 
    - 重構以支援起始地點列表。
    - 使用 `numpy.random.choice` 搭配模型輸出的 `p=probabilities`。

## 2025-12-06 實作與驗證

### 已套用變更 (Changes Applied)
- **機率性預測：** 在 `run_pandemic_simulation` 中實作了 `predict_proba` + `np.random.choice`。加入了微小的平滑因子 (1e-6) 以防止零機率鎖死。
- **活躍傳染源池：** 實作了池機制。模擬現在會從最後 N 個活躍案例中隨機挑選傳播者。
- **多點支援：** 更新 `run_pandemic_simulation` 以接受 `start_locs` 列表。更新 `__main__` 以模擬從 台北（松山）、桃園（機場）和 高雄（小港）同時開始的爆發。
- **移除混亂率：** 移除了 `chaos_level` 參數，因為不再需要。

### 驗證結果 (Verification Results)
- **測試數據：** 由於原始檔案遺失，建立了一個虛擬數據集 (`Age_County_Gender_day_19CoV.csv`) 進行測試。
- **模擬執行：** 成功。生成了 10,000 筆案例。
- **觀察：** 模擬日誌顯示「當前地點」在三個起始區域（台北、桃園、高雄）之間跳動，證實活躍池機制成功維持了多條並行的傳播鏈。地點也在區域內變化，證實機率擴散機制運作正常。

### 已知問題 / 發現的 Bug (Known Issues / Bugs Spotted)
1. **視覺化腳本 (`visualize_pro3-noram.py`)：**
   - **Bug 1 (已修復)：** `Polygon(pts, True)` 在較新版的 matplotlib 會導致 TypeError。已改為 `Polygon(pts, closed=True)`。
   - **Bug 2 (未解)：** 解析 GeoJSON 座標時出現 `ValueError: setting an array element with a sequence`。這顯示腳本預設的多邊形結構與下載的 `taiwan_townships.json` 中的某些複雜幾何形狀不匹配。
   - **狀態：** 視覺化功能目前暫時無法使用，但模擬核心功能運作正常。

### 下一步 (Next Steps)
- 使用者需提供真實的 `Age_County_Gender_day_19CoV.csv` 進行訓練。
- 若需要視覺化，需修復 `visualize_pro3-noram.py` 中的 GeoJSON 解析邏輯。

## 2025-12-06 視覺化修復與生成

### 已修復項目 (Fixed Issues)
1. **GeoJSON 解析錯誤：** 修復了 `visualize_pro3-noram.py` 中的座標解析邏輯。新增了 `extract_polygons` 遞迴函數，能正確處理巢狀結構不一致的多邊形座標，解決了 `ValueError: setting an array element with a sequence` 錯誤。
2. **顏色映射優化：** 引入了對數顏色映射 (Log Color Mapping)。使用 `mcolors.LogNorm` 搭配黑-黃-橘-紅-紫的漸層，解決了初期案例少時顏色不明顯，後期案例多時顏色飽和的問題，更能呈現爆發規模的層次。

### 執行成果 (Results)
- **成功生成動畫：** `simulation_log_color.gif`
- **視覺效果：** 地圖現在能正確渲染，且傳播熱區（台北、桃園、高雄）的擴散動態清晰可見。

## 2025-12-07 專案設定與資料處理 (Project Setup and Data Processing)

### 完成項目 (Completed Items)
1. **資料前處理 (Data Preprocessing)：** 完成了台灣 COVID-19 反事實分析專案的資料前處理工作。
2. **檔案生成 (Generated Files)：** 成功生成了以下處理後的數據檔案：
   - `twcdc_daily_cases_by_location.csv`
   - `simulated_daily_cases_by_location.csv`

### 下一步 (Next Steps)
- 進行模型訓練或進一步分析 (Model training or further analysis)。

## 2025-12-07 機器學習建模 (XGBoost & SHAP)

### 執行項目 (Executed Tasks)
1. **資料準備 (Data Preparation)：**
   - 重新讀取並聚合全台每日「本土」與「境外移入」病例。
   - 確保日期連續性，填補缺失值。

2. **特徵工程 (Feature Engineering)：**
   - **Lags (滯後特徵)：** 過去 1, 3, 7, 10, 14, 21 天的病例數。
   - **Rolling Means (移動平均)：** 過去 7, 14 天的平均與標準差。
   - **Calendar Features (日曆特徵)：** 星期幾 (Day of Week)、月份 (Month)。

3. **模型訓練 (Model Training)：**
   - **模型：** XGBoost Regressor
   - **驗證策略：** Time Series Split (前 80% 訓練，後 20% 測試)，避免未來數據洩漏。

4. **模型解釋 (Model Interpretation)：**
   - **工具：** SHAP (SHapley Additive exPlanations)
   - **修正：** 將 `shap.Explainer` 替換為 `shap.TreeExplainer` 以支援 XGBoost 樹模型。

### 結果摘要 (Results Summary)
- **預測表現：** Test RMSE 約為 1274。模型捕捉到了主要趨勢，但對極端峰值有低估或滯後現象。
- **特徵重要性：**
  - 近期滯後特徵 (Local Lags) 為最強預測因子 (自回歸特性)。
  - 境外移入與長期滯後特徵亦顯示出相關性。

### 下一步 (Next Steps)
- **深度學習 (Deep Learning)：** 推進至 `04_深度學習_時間序列_LSTM.ipynb`，探索 LSTM 是否能提升準確度。
- **流行病學模型 (Epidemiological Models)：** 考慮 `05_流行病學模型_SEIR_ML校準.ipynb` 結合領域知識。