# Oura Ring Health Report Generator

This project converts Oura-exported CSV files into a readable, shareable HTML health report.

---

## English

### Example Report

- Live demo (GitHub Pages):
  https://minniefaye.github.io/oura-health-report-generator/examples/Dec2025_Feb2026.html
- Example file in this repo:
  `examples/Dec2025_Feb2026.html`

This example is for demonstration only.
Please remove or anonymize personal/identifiable information before sharing your own report.

### What You Get

- Trend charts (sleep, activity, heart rate, stress, etc.)
- Risk stratification (red/yellow/green)
- Score-band distribution (with percentages)
- Preventive, education-oriented action suggestions

### How to Export Data from Your Oura Account

1. Log in to Oura and open **Membership Hub**.
2. In account management, click **Export data**.
3. Click **Request your data**.
4. After Oura confirms your export, download the file (usually ZIP).
5. Unzip it and keep all CSV files in one folder.

Tip: Keep each export in a separate folder for clean version tracking.

### Install and Run

Requirements:
- Python 3.9+
- pandas

Install:
```bash
python3 -m pip install -r requirements.txt
```

Generate report:
```bash
python3 generate_report.py --input-dir /path/to/oura-export-folder
```

Optional:
```bash
python3 generate_report.py \
  --input-dir /path/to/oura-export-folder \
  --output-html final_report.html \
  --assets-dir report_assets
```

### Output Files

- `final_report.html`
- `report_assets/` (SVG charts)

### Common Input CSV Names

- `dailyactivity.csv`
- `dailysleep.csv`
- `dailyreadiness.csv`
- `dailyspo2.csv`
- `daytimestress.csv`
- `heartrate.csv`
- `temperature.csv`
- `sleepmodel.csv`

If some files are missing, related sections are skipped and report generation continues.

### Mobile View

- The report is responsive.
- Wide tables are intentionally horizontally scrollable on mobile for readability.

### FAQ

Q1. Why does table look wide on mobile?
A: Some tables are intentionally horizontally scrollable to keep columns readable.

Q2. Can I use this for non-Oura data?
A: Partially. The script expects Oura-like filenames and fields for best results.

Q3. Is this a medical diagnosis tool?
A: No. It is for education and screening support only.

Q4. Can I customize colors/layout?
A: Yes. Edit the CSS block in `generate_report.py`.

### Privacy and Safety

Do not upload personal raw health exports (CSV/PDF/images) to a public repository.

Suggested practice:
- Public repo: code and documentation only
- Private/local storage: personal raw data and generated reports

### Disclaimer

This tool is for health education and screening support only, not for diagnosis, treatment, or emergency decisions.

---

## 繁體中文

### 專案說明

這個專案可把 Oura 匯出的 CSV 資料，轉成一份可分享、易讀的 HTML 健康報告。

### 範例報告

- 線上展示（GitHub Pages）：
  https://minniefaye.github.io/oura-health-report-generator/examples/Dec2025_Feb2026.html
- 本 repo 範例檔案：
  `examples/Dec2025_Feb2026.html`

此範例僅供展示用途。
分享前請先移除或匿名化個人可識別資訊。

### 你會得到什麼

- 趨勢圖（睡眠、活動、心率、壓力等）
- 風險分級（紅／黃／綠）
- 分數區間分佈（含百分比）
- 預防醫學角度的重點建議（教育用途）

### 如何從 Oura 帳號取得資料

1. 登入 Oura 並進入 **Membership Hub**。
2. 在帳號管理中找到 **Export data**。
3. 點選 **Request your data**。
4. 收到可下載通知後下載資料（通常為 ZIP）。
5. 解壓縮後，將所有 CSV 放在同一個資料夾。

建議：每次匯出使用獨立資料夾，方便管理不同時間區間。

### 安裝與執行

需求：
- Python 3.9+
- pandas

安裝：
```bash
python3 -m pip install -r requirements.txt
```

產生報告：
```bash
python3 generate_report.py --input-dir /path/to/oura-export-folder
```

可選參數：
```bash
python3 generate_report.py \
  --input-dir /path/to/oura-export-folder \
  --output-html final_report.html \
  --assets-dir report_assets
```

### 輸出檔案

- `final_report.html`
- `report_assets/`（圖表 SVG）

### 常見輸入檔名

- `dailyactivity.csv`
- `dailysleep.csv`
- `dailyreadiness.csv`
- `dailyspo2.csv`
- `daytimestress.csv`
- `heartrate.csv`
- `temperature.csv`
- `sleepmodel.csv`

若缺少部分檔案，程式會略過對應章節並繼續產生報告。

### 手機版說明

- 報告已做響應式設計。
- 寬表格在手機上可左右滑動（此為預期行為，用來保持欄位可讀性）。

### 常見問題 FAQ

Q1. 為什麼手機上表格看起來很寬？
答：某些表格刻意設計為可左右滑動，以保持欄位可讀性。

Q2. 可以用在非 Oura 資料嗎？
答：可以部分使用，但最佳效果仍需 Oura 風格檔名與欄位。

Q3. 這是醫療診斷工具嗎？
答：不是。此工具僅供健康教育與早期篩檢輔助。

Q4. 可以自訂顏色和排版嗎？
答：可以，直接修改 `generate_report.py` 內的 CSS 區塊即可。

### 隱私與安全

請不要把個人健康原始資料（CSV／PDF／圖片）上傳到公開 GitHub。

建議做法：
- 公開 repo：只放程式與文件
- 私有或本機：保存個人原始資料與產出報告

### 免責聲明

本工具僅供健康教育與早期篩檢輔助，不可作為診斷、治療或急症判斷依據。
