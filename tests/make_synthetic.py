"""Generate synthetic Oura-style CSVs for beta testing (no real health data)."""
import csv, json, math, os, random, sys
from datetime import date, timedelta

random.seed(42)
out = sys.argv[1]
days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
os.makedirs(out, exist_ok=True)
start = date(2026, 11, 1)
dates = [start + timedelta(days=i) for i in range(days)]

def w(name, fields, rows):
    with open(os.path.join(out, name), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows: wr.writerow(r)
    print(name, len(rows))

w("dailyactivity.csv", ["day","score","steps","active_calories","total_calories"],
  [{"day":d,"score":70+int(10*math.sin(i)),"steps":6000+i*137%4000,"active_calories":300+i*7%200,"total_calories":2100+i*11%300} for i,d in enumerate(dates)])
w("dailysleep.csv", ["day","score"],
  [{"day":d,"score":72+int(8*math.cos(i))} for i,d in enumerate(dates)])
w("dailyreadiness.csv", ["day","score","temperature_deviation"],
  [{"day":d,"score":74+int(9*math.sin(i/2)),"temperature_deviation":round(0.1*math.sin(i),2)} for i,d in enumerate(dates)])
w("dailyspo2.csv", ["day","spo2_percentage"],
  [{"day":d,"spo2_percentage":json.dumps({"average":97})} for d in dates])
w("dailystress.csv", ["day","recovery_high","stress_high"],
  [{"day":d,"recovery_high":120,"stress_high":90} for d in dates])
w("sleepmodel.csv", ["day","average_heart_rate","total_sleep_duration","deep_sleep_duration","rem_sleep_duration","light_sleep_duration","awake_time","efficiency","lowest_heart_rate","average_hrv","average_breath","bedtime_start","bedtime_end"],
  [{"day":d,"average_heart_rate":58,"total_sleep_duration":27000,"deep_sleep_duration":5400,"rem_sleep_duration":6000,"light_sleep_duration":15000,"awake_time":600,"efficiency":92,"lowest_heart_rate":52,"average_hrv":45,"average_breath":14.5,"bedtime_start":f"{d}T23:10:00+08:00","bedtime_end":f"{d+timedelta(days=1)}T06:40:00+08:00"} for d in dates])
w("heartrate.csv", ["timestamp","bpm"],
  [{"timestamp":f"{d}T12:{m:02d}:00+08:00","bpm":68+m%12} for d in dates for m in range(0,60,15)])
print("done", out)
