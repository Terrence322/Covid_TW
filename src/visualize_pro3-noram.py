import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import os
import gc
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 讀取地圖 & 自動選用正確欄位 (沿用上一版的成功邏輯)
# ==========================================
def get_geojson_and_detect_keys():
    filename = "taiwan_townships.json"
    if not os.path.exists(filename):
        # 如果沒地圖，嘗試下載 (避免使用者誤刪)
        import requests
        url = "https://raw.githubusercontent.com/ronnywang/twgeojson/master/twtown2010.json"
        try:
            print("地圖檔不存在，嘗試下載...")
            r = requests.get(url)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(r.text)
        except:
            print("錯誤：找不到也無法下載地圖檔。")
            return None, None, None

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    props = data['features'][0]['properties']
    
    # 擴充候選名單，確保能抓到 'county' 和 'town'
    county_candidates = ['county', 'County', 'COUNTYNAME', 'CityName']
    town_candidates = ['town', 'Town', 'TOWNNAME', 'TownName']

    c_key = next((k for k in county_candidates if k in props), None)
    t_key = next((k for k in town_candidates if k in props), None)

    if c_key and t_key:
        return data, c_key, t_key
    else:
        # Fallback
        return data, 'county', 'town'

# ==========================================
# 2. 建立索引與匹配 (沿用上一版)
# ==========================================
def clean_name(name):
    if not isinstance(name, str): return ""
    return name.replace("縣", "").replace("市", "").replace("區", "").replace("鄉", "").replace("鎮", "").strip()

def build_map_index(geojson, c_key, t_key):
    index = {}
    for feature in geojson['features']:
        p = feature['properties']
        c = str(p.get(c_key, ""))
        t = str(p.get(t_key, ""))
        full_name = c + t
        index[(clean_name(c), clean_name(t))] = full_name
    return index

def resolve_location(csv_loc, map_index):
    try:
        raw_county, raw_town = csv_loc.split('_')
    except: return None
    
    c_raw = raw_county.replace('台', '臺')
    c_clean = clean_name(c_raw)
    t_clean = clean_name(raw_town)
    
    # 試試現代名
    match = map_index.get((c_clean, t_clean))
    if match: return match
    
    # 試試舊名
    target_c = c_clean
    if "新北" in c_clean: target_c = "臺北"
    elif "桃園" in c_clean: target_c = "桃園"
    elif "臺中" in c_clean: target_c = "臺中"
    elif "臺南" in c_clean: target_c = "臺南"
    elif "高雄" in c_clean: target_c = "高雄"
    
    return map_index.get((target_c, t_clean))

# ==========================================
# 3. 主程式 (重點優化：對數顏色)
# ==========================================
def run_log_color_pipeline():
    # 1. 準備地圖與資料
    geo, c_key, t_key = get_geojson_and_detect_keys()
    if not geo: return
    if not os.path.exists("simulated_outbreak.csv"):
        print("錯誤：找不到 CSV")
        return
    df = pd.read_csv("simulated_outbreak.csv")
    
    # 2. 匹配
    map_index = build_map_index(geo, c_key, t_key)
    df['Map_Key'] = df['Location'].apply(lambda x: resolve_location(x, map_index))
    df_matched = df.dropna(subset=['Map_Key'])
    
    if df_matched.empty:
        print("錯誤：無資料匹配成功。")
        return

    # 3. 生成動畫 (重點改動區域)
    print("正在生成對數顏色版 GIF...")
    fig, ax = plt.subplots(figsize=(9, 12))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.axis('off')

    patches = []
    patch_indices = []
    for feature in geo['features']:
        p = feature['properties']
        full_name = str(p.get(c_key, "")) + str(p.get(t_key, ""))
        geom = feature['geometry']
        
        # --- [修復] 更強健的座標解析邏輯 ---
        raw_coords = geom['coordinates']
        
        # 遞迴函數：找出所有最底層的 [x, y] 列表
        def extract_polygons(coords):
            # 檢查是否為單個多邊形的座標列表 (通常是 List[List[x,y]])
            # 特徵：第一層是 list, 第二層是 list, 第三層是數字 (x, y)
            # 但 numpy 轉換時若不規則會報錯，所以我們手動判斷深度
            if not isinstance(coords, list): return []
            if not coords: return []
            
            # 判斷是否到了 [[x,y], [x,y], ...] 這一層
            # 檢查第一個元素是否為 [x,y] 形式
            first_elem = coords[0]
            if isinstance(first_elem, list) and len(first_elem) >= 2 and isinstance(first_elem[0], (int, float)):
                return [coords] # 找到一個多邊形環
            
            # 否則繼續遞迴
            polys = []
            for sub in coords:
                polys.extend(extract_polygons(sub))
            return polys

        # 提取出所有的多邊形 (每個多邊形是一組 [[x,y],...])
        all_polys = extract_polygons(raw_coords)
        
        for poly_coords in all_polys:
            pts = np.array(poly_coords)
            if pts.ndim == 2 and len(pts) > 2:
                patches.append(Polygon(pts, closed=True))
                patch_indices.append(full_name)

    p_coll = PatchCollection(patches, alpha=1.0, edgecolor='#333333', facecolor='#111111', linewidth=0.2)
    ax.add_collection(p_coll)
    ax.autoscale_view()
    
    # --- [關鍵優化] 使用對數顏色映射 ---
    # 顏色: 黑 -> 黃 -> 橘 -> 紅 -> 紫
    colors = ["#111111", "#FFFF00", "#FF8800", "#FF0000", "#990099"]
    cmap = mcolors.LinearSegmentedColormap.from_list("PandemicLog", colors, N=256)
    
    # 計算最大值，用於設定顏色範圍
    df_matched['Date'] = pd.to_datetime(df_matched['Date'])
    daily = df_matched.groupby(['Date', 'Map_Key']).size().reset_index(name='New')
    pivot = daily.pivot(index='Date', columns='Map_Key', values='New').fillna(0)
    full_range = pd.date_range(start=pivot.index.min(), end=pivot.index.max())
    pivot = pivot.reindex(full_range, fill_value=0)
    cum_df = pivot.cumsum()
    max_cases = cum_df.max().max()
    
    # 使用 LogNorm，vmin 設為 1 (避免 log(0) 錯誤)，vmax 設為實際最大值
    # 這樣 1~10 的變化會跟 100~1000 的變化在視覺上差不多明顯
    norm = mcolors.LogNorm(vmin=1, vmax=max_cases)

    title_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, color='white', fontsize=16)
    
    # 取樣 (每 3 天一幀，加快動畫速度)
    dates = cum_df.index[::3]
    
    def update(frame_date):
        title_text.set_text(frame_date.strftime('%Y-%m-%d'))
        if frame_date in cum_df.index:
            day_data = cum_df.loc[frame_date]
            new_colors = []
            for name in patch_indices:
                val = day_data.get(name, 0)
                # 只有大於 0 的才套用顏色，否則維持深灰
                new_colors.append(cmap(norm(val)) if val > 0 else '#111111')
            p_coll.set_facecolors(new_colors)
        return p_coll, title_text

    ani = animation.FuncAnimation(fig, update, frames=dates, blit=False, interval=100)
    output = "simulation_log_color.gif"
    ani.save(output, writer='pillow', fps=12) # FPS 提高一點
    print(f"完成！請查看 {output}，顏色層次應該更豐富了。")
    plt.close()
    gc.collect()

if __name__ == "__main__":
    run_log_color_pipeline()