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

### Install as a Muse App

The report can also live as a private Muse app with a history of dated
report snapshots. The app opens on an Oura-style Summary tab (score
rings, vitals, a Stress & Recovery card, top actions), with detail tabs
for Resilience, Comparison, Trends, Notes, and Actions. The complete
app source is in [`app/`](app/), with install instructions in
[`app/INSTALL.md`](app/INSTALL.md). The app template ships with sample
seed data only — a real person's report values, CSVs, and generated
reports must never be committed to this repository.

### Estimated Resilience (in the HTML report)

The Python report (`generate_report.py`) also computes the same transparent
resilience estimate as a report section (section 6), with the contributors
table, Trend and Map charts, and per-day Stress & Recovery bars. Data rule:
all computations use the maximum data available, with a minimum of 7 days.
Daily detail is organized into weekly periods — the default view shows the
past week plus the days of the current period, and earlier weeks are one
click away in the period dropdown. With 7–27 days the estimate runs
provisionally on the 14-day window (labeled "limited history"); with less
than 7 days the section prints a short note and the rest of the report is
unaffected.

### Tabbed layout (HTML report)

The HTML report mirrors the companion app's layout: it opens on a **Summary**
tab (Sleep and Readiness score rings, vitals with risk dots, Needs Attention,
a Stress & Recovery card, and the top two actions), with tabs for Resilience,
Comparison, Trends, Notes, and Actions. Tabs are plain HTML/CSS/JS in the
single output file — no server needed — and printing shows all sections.
The Summary, Comparison, and Trends tabs each carry the same **Period**
selector as the Resilience section (Latest / All data / Week N), recomputing
their rings, vitals, comparison table, band distribution, and trend charts
per period; Notes and Actions always use the full window.

The report is framed around **health resilience** (Lu MC, keynote on Health
Resilience, Taiwan, November 2026 — see the report's References). If a file
named `resilience_background.html` sits in the data folder, the generator
adds its contents to the Notes tab as a "Background: Health Resilience"
section (see `examples/resilience_background.example.html`). Keep
third-party material in your private copy only.

### Estimated Resilience (in the app)

The app's Resilience tab shows an **Estimated Resilience**: a
transparent estimate computed by the user's own Muse from their own
ring data. It is **not Oura's proprietary score** — Oura's algorithm is
not public and its resilience API endpoint is not available on this
connection, so the app computes and labels its own estimate instead.

How it is computed (contributors over 14-day windows, as Oura labels
them):

- **Nighttime recovery** — 70% sleep component (sleep score, minus a
  penalty per hour of sleep below 7h) + 30% physiology component scored
  against the user's own prior-month baselines (HRV above/below the
  baseline raises/lowers it; lowest resting heart rate above the
  baseline lowers it).
- **Daytime recovery** — average daily restorative (recovery) minutes,
  scaled to 0–100.
- **Daytime stress load** — average daily stress minutes, scaled to
  0–100.

The **level index** blends the three contributors (0.4 nighttime +
0.3 daytime recovery + 0.3 × (100 − stress load)), combining the
14-day value (40%) with a 28-day value (60%), because the level moves
more slowly than the contributors. Levels: **Limited** <40,
**Adequate** 40–54, **Solid** 55–64, **Strong** 65–79,
**Exceptional** ≥80.

The **Map** plots each day's 14-day rolling average of stress vs
recovery minutes (not raw daily values) on fixed scales (stress
0–360 min, recovery 0–240 min), with an axis cross and four curved
dashed lines that mark the boundaries between the five levels,
color-coded from Exceptional (green) to Limited (red). A fixed,
disclosed display offset is applied to the recovery axis so the
trajectory registers against the level bands, as on Oura's own map,
whose recovery axis is normalized rather than raw minutes.

The method was calibrated against official Resilience readings in the
Oura app (level and contributor ratings compared at several dates) and
is shown as small print in the app itself. Treat it as an estimate for
personal trend-watching, not a medical measurement.

### For Beta Testers: Install the App in Your Own Muse — Safely

You can run this report as a private app in your own Muse account.
Your health data belongs to you alone: follow these steps exactly, and
neither your data nor your credentials ever need to leave your own
accounts.

**What you need**

- Your own Muse account, your own Oura Ring and Oura account.
- Your own Oura developer application (next step) — never use someone
  else's client ID/secret, and never share yours.

**Install steps (ask your Muse to do them with you)**

1. **Register your own Oura API application** at
   https://cloud.ouraring.com/oauth/applications and set its redirect
   URI to exactly `https://agent.meta.ai/api/hatch/oauth/callback`.
   Keep your client secret in your password manager or Muse's secure
   credential storage — never in the repo, a chat screenshot, or an
   issue.
2. **Connect Oura in your Muse** as a custom connector named
   `custom.oura` (OAuth2; authorize host
   `https://developer.ouraring.com/authorize`, token URL
   `https://api.ouraring.com/oauth/token`, client auth
   `client_secret_post`, scopes `daily heartrate spo2`). Authorize it
   with **your** Oura account.
3. **Build the app** from this repo's [`app/`](app/) folder, following
   [`app/INSTALL.md`](app/INSTALL.md). Your Muse builds it verbatim
   into your account; the app is private to you.
4. **Pull your data and generate your report locally** with
   `build_csvs.py` and `generate_report.py` (or use a CSV export from
   your Oura Membership Hub), then have your Muse seed the app with
   your report through its `seedSnapshot` / `addSnapshot` actions.
5. **Check the app**: it should open on your latest report, with past
   reports kept as dated snapshots.

**Protect your health information — rules for beta testers**

- **Never commit or push your data.** Your CSVs, generated HTML
  reports, screenshots of your report, and any real `seed-data.ts`
  must never be added to this or any public repository, issue, or pull
  request. The repo `.gitignore` already blocks `*.csv`, `*.html`, and
  `report_assets/` — do not remove those rules. Before any push, run
  `git status` and read every file you are about to send.
- **Keep your seeded app copy private.** If you fork this repo, your
  fork's `app/server/src/seed-data.ts` must keep the sample data. Your
  real seed lives only inside your Muse app, never in a repo.
- **File bug reports with synthetic data only.** Use the sample seed or
  the synthetic test generator; describe real-data problems in words,
  without values, dates, or screenshots that reveal your metrics.
- **Your Muse app is private to your account.** Do not publish, share,
  or screenshot-share your app or its contents.
- **You can disconnect at any time.** Revoke the app's access in your
  Oura account settings, and delete the connector and the app in Muse
  if you stop testing. Deleting the app deletes its stored snapshots.
- **If you ever do push health data by mistake:** remove it
  immediately, rewrite the repository history (or delete and recreate
  the fork), and treat any exposed Oura credentials as compromised —
  delete that Oura application and register a new one. Tell the
  maintainer what happened.

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

This tool is for health education and screening support only. It is not a
medical device and does not provide diagnosis, treatment, or emergency
guidance; its output is not medical advice. The developer and contributors
accept no liability for reliance on generated reports. Please read the
full [DISCLAIMER.md](DISCLAIMER.md) before use.

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

### Beta 測試者：在自己的 Muse 安全安裝 App

你也可以把這份報告安裝成自己 Muse 帳號裡的私人 App。請務必照以下
步驟進行，你的健康資料與憑證不需要離開你自己的帳號：

1. **註冊你自己的 Oura API 應用程式**：到
   https://cloud.ouraring.com/oauth/applications 建立，redirect URI
   必須正確設為 `https://agent.meta.ai/api/hatch/oauth/callback`。
   Client secret 只存放在密碼管理器或 Muse 的安全憑證儲存中，絕不
   放進 repo、聊天截圖或 issue，也不要使用別人的 ID/secret。
2. **在你的 Muse 連接 Oura**：自訂連接器 `custom.oura`（OAuth2；
   授權主機 `https://developer.ouraring.com/authorize`、token URL
   `https://api.ouraring.com/oauth/token`、`client_secret_post`、
   scopes `daily heartrate spo2`），並用你自己的 Oura 帳號授權。
3. **建立 App**：請你的 Muse 依照 [`app/INSTALL.md`](app/INSTALL.md)
   以本 repo 的 `app/` 資料夾原樣建立；App 僅屬於你的帳號、對外不公開。
4. **在本機拉取資料並產生報告**（`build_csvs.py` +
   `generate_report.py`，或使用 Oura Membership Hub 匯出的 CSV），
   再由你的 Muse 透過 `seedSnapshot` / `addSnapshot` 把報告放入 App。
5. **確認 App** 能開啟你最新的報告，過往報告以日期快照保存。

保護健康資訊的規則：

- **絕對不要 commit 或 push 你的資料**：CSV、產出的 HTML 報告、報告
  截圖、真實的 `seed-data.ts`，都不得放進任何公開 repo、issue 或 PR。
  本 repo 的 `.gitignore` 已封鎖 `*.csv`、`*.html` 與
  `report_assets/`，請勿移除；每次 push 前先 `git status` 逐一檢查。
- **Fork 的 seed 只能保留範例資料**；你的真實 seed 只存在於你的
  Muse App 內，永遠不進 repo。
- **回報 bug 只用合成資料**（範例 seed 或合成測試產生器）；描述真實
  資料問題時只用文字，不要附數值、日期或能辨識你健康指標的截圖。
- **隨時可以退出測試**：在 Oura 帳號設定撤銷授權，並在 Muse 刪除
  連接器與 App（刪除 App 會一併刪除其中保存的快照）。
- **若不慎 push 了健康資料**：立即移除並重寫 repo 歷史（或刪除重建
  fork），已曝光的 Oura 憑證視為外洩——刪除該應用程式並重新註冊，
  同時告知維護者。

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

本工具僅供健康教育與早期篩檢輔助，不是醫療器材，不提供診斷、治療或
急症判斷依據，產出內容不構成醫療建議；開發者與貢獻者不對信賴產出報告
所生之結果負責。使用前請詳閱完整免責聲明：[DISCLAIMER.md](DISCLAIMER.md)。
