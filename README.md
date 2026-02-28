# Oura 戒指健康報告產生器  
# Oura Ring Health Report Generator

把 Oura 匯出的 CSV 資料，轉成一份可分享、易讀的 HTML 健康報告。  
This tool converts Oura-exported CSV files into a readable, shareable HTML health report.

---

## 你會得到什麼 / What You Get

- 趨勢圖（睡眠、活動、心率、壓力等）
- 風險分級（紅/黃/綠）
- 分數分佈（含百分比）
- 醫學角度的重點建議（教育用途）

---

## 如何從 Oura 帳號取得資料（最重要）  
## How to Export Data from Your Oura Account

### 步驟 / Steps
1. 登入 Oura 官網並進入 **Membership Hub**。  
2. 在帳號管理區找到 **Export data**。  
3. 點選 **Request your data** 送出匯出請求。  
4. 收到 Oura Email 後，回到 Membership Hub 下載資料（通常是 ZIP）。  
5. 解壓縮後，你會看到多個 CSV（例如 `dailyactivity.csv`、`dailysleep.csv` 等）。  

1. Log in to Oura and open **Membership Hub**.  
2. In account management, click **Export data**.  
3. Click **Request your data**.  
4. After Oura confirms your export, download the file (usually ZIP).  
5. Unzip it and keep all CSV files in one folder.

> 建議：每次匯出都放在獨立資料夾，方便追蹤不同時間區間。 Tip: Keep each export in a separate folder for clean version tracking.

---

## 安裝與執行 / Install & Run

### 需求 / Requirements
- Python 3.9+
- pandas

### 安裝 / Install
```bash
python3 -m pip install -r requirements.txt
```

### 產生報告 / Generate Report
```bash
python3 generate_report.py --input-dir /path/to/oura-export-folder
```

### 可選參數 / Optional
```bash
python3 generate_report.py \
  --input-dir /path/to/oura-export-folder \
  --output-html final_report.html \
  --assets-dir report_assets
```

---

## 輸出檔案 / Output Files

- `final_report.html`
- `report_assets/`（圖表 SVG）

---

## 常見輸入檔名 / Common Input CSV Names

- `dailyactivity.csv`
- `dailysleep.csv`
- `dailyreadiness.csv`
- `dailyspo2.csv`
- `daytimestress.csv`
- `heartrate.csv`
- `temperature.csv`
- `sleepmodel.csv`

> 若缺少部分檔案，程式會略過對應章節並繼續產生報告。  
> If some files are missing, related sections are skipped and report generation continues.

---

## 手機版說明 / Mobile View

- 報告已做響應式設計。  
- 寬表格在手機上可左右滑動（這是預期行為）。  

---

## FAQ

Q1. Why does table look wide on mobile?  
Q1. 為什麼手機上表格看起來很寬？  
A: Some tables are intentionally horizontally scrollable to keep columns readable.  
答： 某些表格刻意設計為可左右滑動，以保持欄位可讀性。  

Q2. Can I use this for non-Oura data?  
Q2. 可以用在非 Oura 資料嗎？  
A: Partially. The script expects Oura-like filenames and fields for best results.  
答： 可以部分使用，但最佳效果仍需 Oura 風格檔名與欄位。  

Q3. Is this a medical diagnosis tool?  
Q3. 這是醫療診斷工具嗎？  
A: No. It is for education and screening support only.  
答： 不是。此工具僅供健康教育與早期篩檢輔助。  

Q4. Can I customize colors/layout?  
Q4. 可以自訂顏色和排版嗎？  
A: Yes. Edit the CSS block in `generate_report.py`.  
答： 可以，直接修改 `generate_report.py` 內的 CSS 區塊即可。  

---

## 隱私與安全 / Privacy & Safety

請不要把個人健康原始資料（CSV/PDF/圖片）上傳到公開 GitHub。  
Do not upload personal raw health exports to a public repository.

建議：
- 公開 repo：只放程式與文件  
- 私有/本機：保存個人原始資料與報告

---

## 免責聲明 / Disclaimer

本工具僅供健康教育與早期篩檢輔助，不可作為診斷、治療或急症判斷依據。  

This tool is for health education and screening support only, not for diagnosis, treatment, or emergency decisions.
