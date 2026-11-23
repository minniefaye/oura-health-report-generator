# Install the Oura Health Report as a Muse app

This folder is a complete Muse web app (TypeScript Space) that displays a
person's Oura preventive health report — the same report `generate_report.py`
renders as HTML — as a private, installable app with a history of dated
report snapshots. The app opens on an Oura-style Summary tab, with detail
tabs for Resilience, Comparison, Trends, Notes, and Actions. The
Resilience tab shows the app's own **Estimated Resilience** — a
transparent estimate computed from the user's ring data (see the README
section "Estimated Resilience (in the app)"); it is not Oura's official
score.

## For the installing Muse — read this first

- Build the app from the files in this `app/` folder **verbatim**. Do not
  redesign the UI or restructure the code; every install should look and
  behave identically.
- App name: **Oura Health Report**. Slug: `oura-health-report` (see
  `space.json`).
- The app is **private to the installing user's account**. Never publish
  or share it, and never put the user's health data in this repo.

## Privacy rules (non-negotiable)

- `server/src/seed-data.ts` is the ONLY file that may hold a person's
  report values. The copy in this repo is **sample data with invented
  round numbers** — it is not anyone's real report.
- A real person's seed file, their Oura CSVs, and their generated reports
  must never be committed or pushed anywhere. The repo `.gitignore`
  already excludes `*.csv`, `*.html`, and `report_assets/`.
- Show generated reports only to the person they belong to.

## Install steps

1. **Build the app** from `app/` exactly as shipped. It creates four
   actions: `seedSnapshot`, `listSnapshots`, `getSnapshot`, `addSnapshot`.
2. **Connect the user's Oura account** (needed for real data):
   - Oura Personal Access Tokens were deprecated in Dec 2025; use OAuth2.
   - The user registers their own API application at
     https://cloud.ouraring.com/oauth/applications and connects it as a
     custom connector named `custom.oura` (authorize host
     `https://developer.ouraring.com/authorize`, token URL
     `https://api.ouraring.com/oauth/token`, client auth
     `client_secret_post`, scopes `daily heartrate spo2`).
3. **Pull their data and generate a report** with the tools in the repo
   root:
   ```bash
   python3 build_csvs.py --timezone <their timezone> --output-dir <private dir>/input
   python3 generate_report.py --input-dir <private dir>/input --output-html report.html
   ```
   Or point `generate_report.py` at a folder of CSVs from an Oura
   Membership Hub data export instead of using the API.
4. **Seed the app with their first real report**: convert the generated
   report into the payload shape defined in
   `server/src/report-schema.ts` (the sample seed shows the exact shape),
   replace `server/src/seed-data.ts` with their snapshot **in the local
   app build only — never in the repo**, rebuild, then call
   `seedSnapshot`. It inserts only when no snapshots exist.
   Alternatively skip seeding and call `addSnapshot` with the payload.
5. **Verify**: open the app. It should open on their latest report with
   the comparison table, score bands, notes, seven trend charts,
   recommendation cards, and disclaimer, and list the snapshot in the
   history strip.

## Optional payload fields: stressDays and resilience

Four optional payload fields light up extra UI when present (older
snapshots without them render exactly as before):

- `stressDays` — Oura's per-day stress summary labels (`day_summary` in
  `dailystress.csv`, which `build_csvs.py` now exports). They render as
  day-label chips and feed the Summary tab's Stress & Recovery card.
- `resilience` — the Estimated Resilience payload (level, index,
  contributors, trend, map, method). The installing Muse computes it
  from the user's own pulled ring data using the documented method (see
  the README section "Estimated Resilience (in the app)") and includes
  it in the snapshot payload. The schema requires an `official_anchor`
  object: when calibrating against the user's own Oura app, record their
  official reading there; for a first snapshot without one, use a
  clearly-labeled placeholder. Never put anyone else's readings or
  values in the repo copy — the repo seed's anchor is an invented sample.
  Its optional `dailySeries` field (the full-dataset daily series)
  drives the Resilience tab's own Period dropdown over weekly periods.
- `periodViews` — precomputed per-period views (latest / all data / one
  per weekly period), each with its own summary, comparison, bands and
  charts; when present, the Summary, Comparison and Trends tabs each get
  a Period dropdown over them.

## Refreshing later

The app cannot call Oura itself. When the user asks for a fresh report,
their Muse pulls new data (step 3), generates the report, converts it to
the payload shape, and calls `addSnapshot`. A snapshot with the same data
window is replaced; a new window adds a new history entry. No rebuild is
needed for refreshes.
