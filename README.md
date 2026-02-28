# Oura Health Report Generator  
# Oura 健康報告產生器

Generate a polished preventive-health HTML report from Oura-style CSV exports.  
從 Oura 匯出的 CSV 資料，自動產生一份美觀、可分享的預防醫學 HTML 報告。

---

## 1) What This Project Does / 這個專案能做什麼

**English**
- Scan Oura-style CSV exports recursively.
- Build one HTML report with trend charts, score distribution, and action suggestions.
- Add risk highlighting (red/yellow/green) for faster interpretation.
- Keep report mobile-friendly (responsive layout + scrollable tables).

**繁體中文**
- 遞迴掃描 Oura 風格的 CSV 匯出資料。
- 產生單一 HTML 報告，包含趨勢圖、分數分佈與行動建議。
- 以紅/黃/綠風險色標，提升判讀速度。
- 支援手機閱讀（響應式版面 + 可滑動表格）。

---

## 2) Features / 功能特色

- Bilingual sections (English + 繁體中文)
- Preventive comparison table (general healthy standards vs your data)
- Oura score band distribution with percentages
- Visual risk indicators and priority action cards
- Mobile-friendly report layout
- Works even if some datasets are missing (best-effort generation)

---

## 3) Quick Start / 快速開始

### Requirements / 環境需求
- Python 3.9+
- pandas

### Install / 安裝
```bash
python3 -m pip install -r requirements.txt
```

### Run / 執行
```bash
python3 generate_report.py --input-dir /path/to/oura-export
```

### Optional Arguments / 可選參數
```bash
python3 generate_report.py \
  --input-dir /path/to/oura-export \
  --output-html final_report.html \
  --assets-dir report_assets
```

---

## 4) Output / 輸出結果

- `final_report.html` (or your custom `--output-html`)
- `report_assets/` (or your custom `--assets-dir`, contains SVG charts)

Example:
- `/path/to/oura-export/final_report.html`
- `/path/to/oura-export/report_assets/`

---

## 5) Expected Input Files / 預期資料檔名

Auto-detected by filename (case-insensitive).  
依檔名自動偵測（不分大小寫）。

Common files:
- `dailyactivity.csv`
- `dailysleep.csv`
- `dailyreadiness.csv`
- `dailyspo2.csv`
- `daytimestress.csv`
- `heartrate.csv`
- `temperature.csv`
- `sleepmodel.csv`

If a file is missing, related sections are skipped and report generation continues.  
若缺少部分檔案，會略過對應章節，不影響整體報告產生。

---

## 6) Screenshot Section / 截圖區（可放 GitHub 圖片）

You can add screenshots after pushing your repo:

```md
![Report Hero](docs/images/report-hero.png)
![Comparison Table](docs/images/report-table.png)
![Trend Charts](docs/images/report-charts.png)
```

建議你在 repo 裡建立：
- `docs/images/report-hero.png`
- `docs/images/report-table.png`
- `docs/images/report-charts.png`

---

## 7) FAQ

### Q1. Why does table look wide on mobile?  
### Q1. 為什麼手機上表格看起來很寬？
**A:** Some tables are intentionally horizontally scrollable to keep columns readable.  
**答：** 某些表格刻意設計為可左右滑動，以保持欄位可讀性。

### Q2. Can I use this for non-Oura data?  
### Q2. 可以用在非 Oura 資料嗎？
**A:** Partially. The script expects Oura-like filenames and fields for best results.  
**答：** 可以部分使用，但最佳效果仍需 Oura 風格檔名與欄位。

### Q3. Is this a medical diagnosis tool?  
### Q3. 這是醫療診斷工具嗎？
**A:** No. It is for education and screening support only.  
**答：** 不是。此工具僅供健康教育與早期篩檢輔助。

### Q4. Can I customize colors/layout?  
### Q4. 可以自訂顏色和排版嗎？
**A:** Yes. Edit the CSS block in `generate_report.py`.  
**答：** 可以，直接修改 `generate_report.py` 內的 CSS 區塊即可。

---

## 8) Privacy / 隱私提醒

Do **not** upload personal health exports to a public repository.  
請勿將個人健康原始資料上傳到公開 GitHub repo。

Recommended:
- Public repo: code + docs only
- Private/local: raw CSV/PDF/images and generated personal reports

---

## 9) Disclaimer / 免責聲明

This tool is for health education and screening support only.  
It is **not** a diagnosis, treatment plan, or emergency assessment.

本工具僅供健康教育與早期篩檢輔助，  
**不可**作為診斷、治療或急症判斷依據。
