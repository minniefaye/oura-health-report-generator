# Beta test notes

Tested at commit 90b9b9e with synthetic Oura-style data (no real health data):
full 14-day set, 30-day set, single day, partial files, empty folder,
header-only / malformed / empty CSVs. Python 3.12, pandas 3.1.1 (venv).

## Bugs found and fixed in this working copy

1. **Crash on partial columns (generate_report.py)** — the README promises
   missing files just skip their sections, but a present file with a missing
   *column* crashed the whole run with KeyError:
   - `dailystress.csv` with `stress_high` but no `recovery_high`
   - `sleepmodel.csv` with `deep_sleep_duration` but no `rem_sleep_duration`
   - `daytimestress.csv` with only one of `stress_value` / `recovery_value`
   All three chart blocks now require both columns before running.

2. **No feedback about missing inputs (generate_report.py)** — the generator
   now prints which CSVs were found, warns clearly on an empty input folder,
   and lists missing files whose sections will be skipped.

3. **build_csvs.py only worked on the author's machine** —
   - Hard-coded Muse credential helper: now falls back cleanly, and an
     `OURA_TOKEN` environment variable (Oura Personal Access Token) works
     anywhere. A missing/unavailable Muse credential now exits with a
     plain-language message instead of a traceback.
   - Hard-coded two-week window: default is now the last 14 days.
   - Hard-coded output folder and Los Angeles timezone: new
     `--output-dir` and `--timezone` options (defaults unchanged).

4. **HTTP 422 on heartrate for UTC+ timezones (build_csvs.py)** — found
   on the first live pull in a UTC+ timezone: the query string was
   built by hand, so the `+` in `+08:00` was read as a space and Oura
   rejected the datetime. A `-` offset (the author's Los Angeles) never
   triggered it. Fixed by building the query with `urllib.parse.urlencode`;
   heartrate then returned samples normally for the full window.

5. **Setup note (not a code bug)** — Oura deprecated Personal Access Tokens
   in Dec 2025, so the token path above is OAuth-only in practice. Also,
   Oura's newer authorize host `developer.ouraring.com/authorize` worked
   for a newly registered application where `cloud.ouraring.com/oauth/authorize`
   returned 400 invalid_request.

## Verified working
- Full / partial / empty / single-day / malformed inputs all produce a report.
- Example report in `examples/` matches what the generator produces.
- Bad API token: endpoints warn HTTP 401 and write header-only CSVs (no crash).

## Suggestions not yet implemented (author's call)
- The upstream description mentions user-selectable English-only vs bilingual
  output, but the generator has no `--lang` option — output is always bilingual.
- `tests/make_synthetic.py` (added here) generates fake Oura CSVs so future
  changes can be regression-tested without anyone's real health data.
