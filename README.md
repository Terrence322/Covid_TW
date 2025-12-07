# 未曾發生的疫情：臺灣 COVID-19 反事實模擬與因果推論

一個 **「動機＋背景＋資料＋模型＋為什麼」** 的完整資料科學專案架構。

---

## 1. 動機（Motivation）

COVID-19 疫情期間，臺灣 2020–2021 的死亡率在全球算是非常低的一群，
很多人只印象「臺灣守得很好」，但很少有人真正量化：

> **如果當初邊境沒有守好、隔離沒有落實，整個 2018–2022 的死亡與感染結構，
> 會跟現在差多少？**

這個 Side Project 想做的，不只是畫一條漂亮的預測線，而是：

1. **用 TWCDC 的真實資料建模，還原「臺灣到底靠什麼因素穩住疫情」。**
2. **用機器學習 + 流行病學結構模型，模擬一個「未曾發生的疫情平行世界」。**
3. **把這個平行世界跟真實歷史比較：死亡數、感染數、醫療負擔差多少。**

---

## 2. 背景與研究問題（Background & Research Questions）

### 背景：臺灣在 COVID 疫情中的特殊性

* 2020–2021：臺灣靠著 **嚴格邊境管制 + 隔離 + NPI（口罩、實名制）**，
  避過了很多國家早期那一波巨量感染與死亡。
* 直到 2022 Omicron 之後才開始逐步開放、與病毒共存。

因此，臺灣是一個很適合問這個問題的 case：

> **如果我們把「守得很好」這件事拿掉，會發生什麼事？**

### 核心問題

1. **解釋性問題（Explaining the past）**
   * 在 2020–2022 期間，本土確診數是如何受到 **境外移入、邊境管制、國內警戒、疫苗覆蓋率、人口移動** 的影響？
   * 哪些因素在模型裡看起來真的有「壓病例」的效果？

2. **反事實問題（Counterfactual）**
   * 如果邊境不那麼嚴格，或境外移入案例沒有被好好隔離，本土確診曲線會變成怎樣？
   * 2020–2022 的 COVID 死亡數會多多少？

3. **方法論問題（Methodological）**
   * 傳統迴歸（GLM）與各種 ML / 深度學習 / 因果方法，哪一些比較擅長預測？哪一些比較擅長回答「如果…會怎樣」的因果問題？

---

## 3. 資料來源與變數設計（Data & Features）

檔案位於 `data/raw/`：

### 3.1 TWCDC COVID 通報資料
* **每日 / 每週分解後的：** 本土病例數、境外移入病例數
* 必要欄位：`date`、`local_cases`、`imported_cases`

### 3.2 政策與 NPI 資料
* **國際資料：** Oxford COVID-19 Government Response Tracker (Stringency Index)
* **手刻變數：** `border_level`（邊境管制等級）、`alert_level`（國內警戒等級）

### 3.3 疫苗資料
* `vax_1dose`、`vax_full`、`vax_booster`

### 3.4 移動與人口資料（選配）
* Google / Apple mobility index
* 縣市人口、年齡結構

---

## 4. 方法架構總覽（Methods Overview）

本專案包含以下分析模組（對應 `notebooks/`）：

1. **Baseline 統計模型**：Generalized Linear Models (Poisson/NB)
2. **機器學習與深度學習**：XGBoost, LSTM/GRU
3. **結構＋因果模型**：SEIR + ML 校準, Synthetic Control

---

## 5. 檔案結構 (Project Structure)

```
.
├── data/
│   ├── raw/                # 原始資料 (Original data)
│   ├── processed/          # 清理後資料 (Processed data)
│   └── external/           # 外部資料 (External: Mobility, Stringency)
├── notebooks/              # Jupyter Notebooks (分析步驟)
│   ├── 01_資料探索與前處理.ipynb
│   ├── 02_統計模型_GLM_Baseline.ipynb
│   ├── 03_機器學習_XGBoost_FeatureImportance_SHAP.ipynb
│   ├── 04_深度學習_時間序列_LSTM.ipynb
│   ├── 05_流行病學模型_SEIR_ML校準.ipynb
│   ├── 06_因果推論_合成對照組_SyntheticControl.ipynb
│   └── 07_總結_反事實模擬與平行世界.ipynb
├── src/                    # 程式碼模組 (Source code)
├── reports/                # 產出報告 (Reports)
└── README.md               # 本文件
```
