"""Build Oura-export-style CSVs from the live API for the report generator.

Default window: 2026-10-27 to 2026-11-09 (same as the two-week analysis).
Pass --start YYYY-MM-DD --end YYYY-MM-DD to build another window.
Writes into ~/workspace/oura-report/input/:
  dailyreadiness.csv  (day, score, temperature_deviation)
  dailysleep.csv       (day, score)
  dailyactivity.csv    (day, score, steps, active_calories, total_calories)
  dailyspo2.csv        (day, spo2_percentage as JSON string)
  dailystress.csv      (day, recovery_high, stress_high)
  sleepmodel.csv       (day, average_heart_rate, total_sleep_duration,
                        bedtime_start, bedtime_end)
  heartrate.csv        (timestamp, bpm) - 5-min samples for the HR chart
No temperature.csv / daytimestress.csv: the v2 API has no matching series,
so those charts are skipped by the generator unless an export CSV is
copied into the input dir first.
"""
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request, read_json_response

BASE = "https://api.ouraring.com"
USER_AGENT = "MinnieOuraBriefing/1.0"
OUT = os.path.expanduser("~/workspace/oura-report/input")
START = "2026-10-27"
END = "2026-11-09"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def get(path, params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE}{path}?{qs}"
    out = []
    while url:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", USER_AGENT)
        add_surrogate_to_request(req, "custom.oura", allowed_hosts=["api.ouraring.com"])
        try:
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=60) as resp:
                body = read_json_response(resp)
        except urllib.error.HTTPError as e:
            print(f"WARN {path} HTTP {e.code}", file=sys.stderr)
            return out
        out.extend(body.get("data") or [])
        nt = body.get("next_token")
        url = f"{BASE}{path}?{qs}&next_token={nt}" if nt else None
    return out


def _tz_offset(d, tz):
    """UTC offset string (e.g. -07:00) for a date in the given timezone."""
    from datetime import datetime

    dt = datetime(d.year, d.month, d.day, 12, 0, tzinfo=tz)
    off = dt.utcoffset()
    total = int(off.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


def write_csv(name, rows, fields):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"{name}: {len(rows)} rows")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()
    start, end = args.start, args.end
    p = {"start_date": start, "end_date": end}
    readiness = get("/v2/usercollection/daily_readiness", p)
    d_sleep = get("/v2/usercollection/daily_sleep", p)
    activity = get("/v2/usercollection/daily_activity", p)
    spo2 = get("/v2/usercollection/daily_spo2", p)
    stress = get("/v2/usercollection/daily_stress", p)
    sleeps = get("/v2/usercollection/sleep", p)
    # Heartrate samples for the whole window (datetime range, LA tz).
    tz = ZoneInfo("America/Los_Angeles")
    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)
    start_off = _tz_offset(start_dt, tz)
    end_off = _tz_offset(end_dt, tz)
    hr = get(
        "/v2/usercollection/heartrate",
        {
            "start_datetime": f"{start}T00:00:00{start_off}",
            "end_datetime": f"{end}T23:59:59{end_off}",
        },
    )

    write_csv(
        "dailyreadiness.csv",
        [{"day": r["day"], "score": r.get("score"), "temperature_deviation": r.get("temperature_deviation")} for r in readiness],
        ["day", "score", "temperature_deviation"],
    )
    write_csv(
        "dailysleep.csv",
        [{"day": d["day"], "score": d.get("score")} for d in d_sleep],
        ["day", "score"],
    )
    write_csv(
        "dailyactivity.csv",
        [
            {
                "day": a["day"],
                "score": a.get("score"),
                "steps": a.get("steps"),
                "active_calories": a.get("active_calories"),
                "total_calories": a.get("total_calories"),
            }
            for a in activity
        ],
        ["day", "score", "steps", "active_calories", "total_calories"],
    )
    write_csv(
        "dailyspo2.csv",
        [
            {
                "day": s["day"],
                "spo2_percentage": json.dumps(s.get("spo2_percentage") or {}),
            }
            for s in spo2
        ],
        ["day", "spo2_percentage"],
    )
    write_csv(
        "dailystress.csv",
        [
            {
                "day": s["day"],
                "recovery_high": s.get("recovery_high"),
                "stress_high": s.get("stress_high"),
            }
            for s in stress
        ],
        ["day", "recovery_high", "stress_high"],
    )
    write_csv(
        "sleepmodel.csv",
        [
            {
                "day": s["day"],
                "average_heart_rate": s.get("average_heart_rate"),
                "total_sleep_duration": s.get("total_sleep_duration"),
                "deep_sleep_duration": s.get("deep_sleep_duration"),
                "rem_sleep_duration": s.get("rem_sleep_duration"),
                "light_sleep_duration": s.get("light_sleep_duration"),
                "awake_time": s.get("awake_time"),
                "efficiency": s.get("efficiency"),
                "latency": s.get("latency"),
                "lowest_heart_rate": s.get("lowest_heart_rate"),
                "average_hrv": s.get("average_hrv"),
                "average_breath": s.get("average_breath"),
                "bedtime_start": s.get("bedtime_start"),
                "bedtime_end": s.get("bedtime_end"),
            }
            for s in sleeps
        ],
        ["day", "average_heart_rate", "total_sleep_duration", "deep_sleep_duration", "rem_sleep_duration", "light_sleep_duration", "awake_time", "efficiency", "latency", "lowest_heart_rate", "average_hrv", "average_breath", "bedtime_start", "bedtime_end"],
    )
    if sleeps:
        print("sleep keys:", sorted(sleeps[0].keys()))
    write_csv(
        "heartrate.csv",
        [{"timestamp": h.get("timestamp"), "bpm": h.get("bpm")} for h in hr],
        ["timestamp", "bpm"],
    )


if __name__ == "__main__":
    main()
