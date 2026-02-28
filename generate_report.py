from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

BASE = Path.cwd()
ASSET_DIR: Path | None = None


def detect_sep(path: Path) -> str:
    first_line = path.read_text(errors='ignore').splitlines()[0]
    return ';' if first_line.count(';') >= first_line.count(',') else ','


def read_csv_robust(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=detect_sep(path), engine='python', on_bad_lines='skip')


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')


def parse_json_field(v: Any) -> Any:
    if not isinstance(v, str):
        return None
    v = v.strip()
    if not v.startswith('{'):
        return None
    try:
        return json.loads(v)
    except Exception:
        return None


def _scale(vals: list[float], out_min: float, out_max: float) -> list[float]:
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [(out_min + out_max) / 2 for _ in vals]
    return [out_min + (v - lo) * (out_max - out_min) / (hi - lo) for v in vals]


def line_svg(xs: list[str], ys: list[float], title: str, color: str) -> str:
    if not ys:
        return '<p class="muted">No chart data.</p>'
    width, height, margin = 860, 280, 36
    chart_w, chart_h = width - margin * 2, height - margin * 2

    x_pix = _scale(list(range(len(ys))), margin, margin + chart_w)
    y_pix = _scale(ys, margin + chart_h, margin)
    points = ' '.join([f"{x:.1f},{y:.1f}" for x, y in zip(x_pix, y_pix)])

    tick_n = 6
    x_tick_lines = []
    y_tick_lines = []
    x_tick_labels = []
    y_tick_labels = []

    for i in range(tick_n):
        tx = margin + chart_w * i / (tick_n - 1)
        x_tick_lines.append(f'<line x1="{tx:.1f}" y1="{margin + chart_h}" x2="{tx:.1f}" y2="{margin + chart_h + 5}" stroke="#cfd8e3"/>')
        idx = int(round((len(xs) - 1) * i / (tick_n - 1))) if xs else 0
        lbl = xs[idx] if xs else ''
        x_tick_labels.append(f'<text x="{tx:.1f}" y="{height - 8}" text-anchor="middle" font-size="10" fill="#5f6c7b">{lbl}</text>')

    y_min, y_max = min(ys), max(ys)
    for i in range(tick_n):
        ty = margin + chart_h * i / (tick_n - 1)
        y_tick_lines.append(f'<line x1="{margin - 5}" y1="{ty:.1f}" x2="{margin}" y2="{ty:.1f}" stroke="#cfd8e3"/>')
        y_val = y_max - (y_max - y_min) * i / (tick_n - 1)
        y_tick_labels.append(f'<text x="{margin - 8}" y="{ty + 3:.1f}" text-anchor="end" font-size="10" fill="#5f6c7b">{y_val:.1f}</text>')

    return f"""
<svg viewBox=\"0 0 {width} {height}\" xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" aria-label=\"{title}\">
  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#fff\" rx=\"12\"/>
  <line x1=\"{margin}\" y1=\"{margin}\" x2=\"{margin}\" y2=\"{margin + chart_h}\" stroke=\"#cfd8e3\"/>
  <line x1=\"{margin}\" y1=\"{margin + chart_h}\" x2=\"{margin + chart_w}\" y2=\"{margin + chart_h}\" stroke=\"#cfd8e3\"/>
  {''.join(y_tick_lines)}
  {''.join(x_tick_lines)}
  <polyline points=\"{points}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"3\"/>
  <text x=\"{margin}\" y=\"20\" font-size=\"12\" fill=\"#5f6c7b\">{title}</text>
  {''.join(y_tick_labels)}
  {''.join(x_tick_labels)}
  <text x=\"{margin + 6}\" y=\"{margin + 14}\" font-size=\"11\" fill=\"#5f6c7b\">max {max(ys):.2f}</text>
  <text x=\"{margin + 6}\" y=\"{margin + chart_h - 6}\" font-size=\"11\" fill=\"#5f6c7b\">min {min(ys):.2f}</text>
</svg>
"""


def dual_line_svg(xs: list[str], y1: list[float], y2: list[float], title: str) -> str:
    pairs = [(x, a, b) for x, a, b in zip(xs, y1, y2) if not (pd.isna(a) and pd.isna(b))]
    if not pairs:
        return '<p class="muted">No chart data.</p>'

    xs2 = [p[0] for p in pairs]
    a = [float(p[1]) if not pd.isna(p[1]) else math.nan for p in pairs]
    b = [float(p[2]) if not pd.isna(p[2]) else math.nan for p in pairs]
    vals = [v for v in a + b if not math.isnan(v)]
    if not vals:
        return '<p class="muted">No chart data.</p>'

    width, height, margin = 860, 280, 36
    chart_w, chart_h = width - margin * 2, height - margin * 2

    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1

    def ypix(v: float) -> float:
        return margin + chart_h - (v - lo) * chart_h / (hi - lo)

    x_pix = _scale(list(range(len(xs2))), margin, margin + chart_w)
    p1 = ' '.join([f"{x:.1f},{ypix(v):.1f}" for x, v in zip(x_pix, a) if not math.isnan(v)])
    p2 = ' '.join([f"{x:.1f},{ypix(v):.1f}" for x, v in zip(x_pix, b) if not math.isnan(v)])

    tick_n = 6
    x_tick_lines = []
    y_tick_lines = []
    x_tick_labels = []
    y_tick_labels = []

    for i in range(tick_n):
        tx = margin + chart_w * i / (tick_n - 1)
        x_tick_lines.append(f'<line x1="{tx:.1f}" y1="{margin + chart_h}" x2="{tx:.1f}" y2="{margin + chart_h + 5}" stroke="#cfd8e3"/>')
        idx = int(round((len(xs2) - 1) * i / (tick_n - 1))) if xs2 else 0
        lbl = xs2[idx] if xs2 else ''
        x_tick_labels.append(f'<text x="{tx:.1f}" y="{height - 8}" text-anchor="middle" font-size="10" fill="#5f6c7b">{lbl}</text>')

    for i in range(tick_n):
        ty = margin + chart_h * i / (tick_n - 1)
        y_tick_lines.append(f'<line x1="{margin - 5}" y1="{ty:.1f}" x2="{margin}" y2="{ty:.1f}" stroke="#cfd8e3"/>')
        y_val = hi - (hi - lo) * i / (tick_n - 1)
        y_tick_labels.append(f'<text x="{margin - 8}" y="{ty + 3:.1f}" text-anchor="end" font-size="10" fill="#5f6c7b">{y_val:.1f}</text>')

    return f"""
<svg viewBox=\"0 0 {width} {height}\" xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" aria-label=\"{title}\">
  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#fff\" rx=\"12\"/>
  <line x1=\"{margin}\" y1=\"{margin}\" x2=\"{margin}\" y2=\"{margin + chart_h}\" stroke=\"#cfd8e3\"/>
  <line x1=\"{margin}\" y1=\"{margin + chart_h}\" x2=\"{margin + chart_w}\" y2=\"{margin + chart_h}\" stroke=\"#cfd8e3\"/>
  {''.join(y_tick_lines)}
  {''.join(x_tick_lines)}
  <polyline points=\"{p1}\" fill=\"none\" stroke=\"#ff7f50\" stroke-width=\"3\"/>
  <polyline points=\"{p2}\" fill=\"none\" stroke=\"#52b788\" stroke-width=\"3\"/>
  <text x=\"{margin}\" y=\"20\" font-size=\"12\" fill=\"#5f6c7b\">{title}</text>
  {''.join(y_tick_labels)}
  {''.join(x_tick_labels)}
  <rect x=\"{margin + chart_w - 170}\" y=\"{margin + 8}\" width=\"10\" height=\"10\" fill=\"#ff7f50\"/><text x=\"{margin + chart_w - 155}\" y=\"{margin + 17}\" font-size=\"11\" fill=\"#5f6c7b\">Stress</text>
  <rect x=\"{margin + chart_w - 90}\" y=\"{margin + 8}\" width=\"10\" height=\"10\" fill=\"#52b788\"/><text x=\"{margin + chart_w - 75}\" y=\"{margin + 17}\" font-size=\"11\" fill=\"#5f6c7b\">Recovery</text>
</svg>
"""


def save_chart(name: str, svg: str) -> str:
    if ASSET_DIR is None:
        raise RuntimeError('ASSET_DIR is not initialized.')
    p = ASSET_DIR / name
    p.write_text(svg, encoding='utf-8')
    return p.name


def profile(series: pd.Series) -> dict[str, float]:
    s = to_num(series).dropna()
    if s.empty:
        return {}
    return {
        'n': float(s.shape[0]),
        'mean': float(s.mean()),
        'median': float(s.median()),
        'p10': float(s.quantile(0.10)),
        'p90': float(s.quantile(0.90)),
        'min': float(s.min()),
        'max': float(s.max()),
    }


def fmt(v: float | None, d: int = 2) -> str:
    if v is None:
        return 'N/A'
    return f"{v:.{d}f}"


def fmt_profile(p: dict[str, float], digits: int = 2) -> str:
    if not p:
        return 'N/A'
    return f"Median {p['median']:.{digits}f} (P10-P90: {p['p10']:.{digits}f}-{p['p90']:.{digits}f})"


def red(text: str) -> str:
    return f'<span class="danger">{text}</span>'


def maybe_red(condition: bool, text: str) -> str:
    return red(text) if condition else text


def color_text(level: str, text: str) -> str:
    if level == 'red':
        return f'<span class="danger">{text}</span>'
    if level == 'yellow':
        return f'<span class="warn">{text}</span>'
    return f'<span class="good">{text}</span>'


def risk_badge(level: str) -> str:
    if level == 'red':
        return '<span class="badge danger">High Risk / 高風險</span>'
    if level == 'yellow':
        return '<span class="badge warn">Mild Caution / 輕度注意</span>'
    return '<span class="badge good">Good / 穩定</span>'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate preventive-health HTML report from Oura-style CSV exports.')
    p.add_argument('--input-dir', default='.', help='Folder containing exported datasets (searched recursively).')
    p.add_argument('--output-html', default='final_report.html', help='Output HTML filename.')
    p.add_argument('--assets-dir', default='report_assets', help='Folder for generated chart assets.')
    return p.parse_args()


def main() -> None:
    global BASE, ASSET_DIR
    args = parse_args()
    BASE = Path(args.input_dir).expanduser().resolve()
    ASSET_DIR = BASE / args.assets_dir
    ASSET_DIR.mkdir(exist_ok=True)

    # Load data
    dataframes: dict[str, pd.DataFrame] = {}
    for path in sorted(BASE.glob('**/*.csv')):
        if args.assets_dir in str(path):
            continue
        rel = str(path.relative_to(BASE))
        try:
            dataframes[rel] = read_csv_robust(path)
        except Exception:
            dataframes[rel] = pd.DataFrame()

    def df_by_filename(filename: str) -> pd.DataFrame:
        filename = filename.lower()
        for rel, df in dataframes.items():
            if Path(rel).name.lower() == filename:
                return df
        return pd.DataFrame()

    dailyactivity = df_by_filename('dailyactivity.csv')
    dailysleep = df_by_filename('dailysleep.csv')
    readiness = df_by_filename('dailyreadiness.csv')
    dailyspo2 = df_by_filename('dailyspo2.csv')
    dailystress = df_by_filename('dailystress.csv')
    daystress = df_by_filename('daytimestress.csv')
    heartrate = df_by_filename('heartrate.csv')
    temperature = df_by_filename('temperature.csv')
    battery = df_by_filename('ringbatterylevel.csv')
    sleepmodel = df_by_filename('sleepmodel.csv')

    # Data window (dynamic)
    all_dates: list[pd.Series] = []
    for df in [dailyactivity, dailysleep, readiness, dailyspo2, dailystress, daystress, heartrate, temperature, battery, sleepmodel]:
        for col in ['day', 'timestamp', 'ts', 'bedtime_start', 'bedtime_end']:
            if col in df.columns:
                all_dates.append(pd.to_datetime(df[col], errors='coerce', utc=True))
    window_text = 'N/A'
    if all_dates:
        merged = pd.concat(all_dates, ignore_index=True).dropna()
        if not merged.empty:
            window_text = f"{merged.min().date()} to {merged.max().date()}"

    # Type conversion
    for df, c in [
        (dailyactivity, 'day'), (dailysleep, 'day'), (readiness, 'day'), (dailyspo2, 'day'), (dailystress, 'day')
    ]:
        if c in df.columns:
            df['day_dt'] = pd.to_datetime(df[c], errors='coerce')

    for df, c in [(heartrate, 'timestamp'), (temperature, 'timestamp'), (battery, 'timestamp'), (daystress, 'timestamp')]:
        if c in df.columns:
            df['ts'] = pd.to_datetime(df[c], errors='coerce', utc=True)

    for df, cols in [
        (dailyactivity, ['steps', 'score', 'active_calories', 'total_calories']),
        (dailysleep, ['score']),
        (readiness, ['score', 'temperature_deviation']),
        (dailyspo2, []),
        (dailystress, ['recovery_high', 'stress_high']),
        (daystress, ['stress_value', 'recovery_value']),
        (heartrate, ['bpm']),
        (temperature, ['skin_temp']),
        (battery, ['level']),
        (sleepmodel, ['average_heart_rate', 'total_sleep_duration']),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = to_num(df[c])

    if 'spo2_percentage' in dailyspo2.columns:
        dailyspo2['spo2_avg'] = to_num(dailyspo2['spo2_percentage'].apply(lambda x: (parse_json_field(x) or {}).get('average')))

    # Derived series
    nightly_hr = sleepmodel.loc[sleepmodel.get('average_heart_rate', pd.Series(dtype=float)) > 0, 'average_heart_rate'] if 'average_heart_rate' in sleepmodel.columns else pd.Series(dtype=float)
    main_sleep_hours = pd.Series(dtype=float)
    if 'total_sleep_duration' in sleepmodel.columns:
        main_sleep_hours = (sleepmodel.loc[sleepmodel['total_sleep_duration'] >= 10800, 'total_sleep_duration'] / 3600.0).dropna()

    sp_steps = profile(dailyactivity['steps']) if 'steps' in dailyactivity.columns else {}
    sp_sleep_score = profile(dailysleep['score']) if 'score' in dailysleep.columns else {}
    sp_readiness = profile(readiness['score']) if 'score' in readiness.columns else {}
    sp_activity_score = profile(dailyactivity['score']) if 'score' in dailyactivity.columns else {}
    sp_spo2 = profile(dailyspo2['spo2_avg']) if 'spo2_avg' in dailyspo2.columns else {}
    sp_nightly_hr = profile(nightly_hr)
    sp_temp_dev = profile(readiness['temperature_deviation']) if 'temperature_deviation' in readiness.columns else {}
    sp_sleep_hours = profile(main_sleep_hours)

    # Out-of-range rates
    sleep_under7 = float((main_sleep_hours < 7).mean() * 100) if not main_sleep_hours.empty else math.nan
    spo2_below95 = float((dailyspo2['spo2_avg'] < 95).mean() * 100) if 'spo2_avg' in dailyspo2.columns and not dailyspo2['spo2_avg'].dropna().empty else math.nan
    nightly_hr_out = float(((nightly_hr < 60) | (nightly_hr > 100)).mean() * 100) if not nightly_hr.empty else math.nan
    temp_dev_over = float((readiness['temperature_deviation'].abs() > 0.5).mean() * 100) if 'temperature_deviation' in readiness.columns and not readiness['temperature_deviation'].dropna().empty else math.nan

    # Score band distributions (Oura 0-100 bands)
    def band_counts(s: pd.Series) -> dict[str, int]:
        x = to_num(s).dropna()
        return {
            'Optimal (85-100)': int((x >= 85).sum()),
            'Good (70-84)': int(((x >= 70) & (x <= 84)).sum()),
            'Fair (60-69)': int(((x >= 60) & (x <= 69)).sum()),
            'Pay attention (0-59)': int((x <= 59).sum()),
            'Total days': int(x.shape[0]),
        }

    band_rows = []
    if 'score' in dailyactivity.columns:
        b = band_counts(dailyactivity['score'])
        b['Score Type'] = 'Activity Score / 活動分數'
        band_rows.append(b)
    if 'score' in dailysleep.columns:
        b = band_counts(dailysleep['score'])
        b['Score Type'] = 'Sleep Score / 睡眠分數'
        band_rows.append(b)
    if 'score' in readiness.columns:
        b = band_counts(readiness['score'])
        b['Score Type'] = 'Readiness Score / 準備度分數'
        band_rows.append(b)
    bands_df = pd.DataFrame(band_rows)
    if not bands_df.empty:
        bands_df = bands_df[['Score Type', 'Optimal (85-100)', 'Good (70-84)', 'Fair (60-69)', 'Pay attention (0-59)', 'Total days']]
        format_cols = ['Optimal (85-100)', 'Good (70-84)', 'Fair (60-69)', 'Pay attention (0-59)', 'Total days']
        score_cols = ['Optimal (85-100)', 'Good (70-84)', 'Fair (60-69)', 'Pay attention (0-59)']
        bands_df[format_cols] = bands_df[format_cols].astype(object)
        for _, row in bands_df.iterrows():
            total = row['Total days'] if row['Total days'] else 0
            score_pcts: dict[str, float] = {}
            for col in format_cols:
                count = int(row[col])
                pct = (count / total * 100.0) if total else 0.0
                bands_df.loc[row.name, col] = f"{count} ({pct:.1f}%)"
                if col in score_cols:
                    score_pcts[col] = pct
            if score_pcts:
                max_col = max(score_pcts, key=score_pcts.get)
                if max_col == 'Pay attention (0-59)':
                    bands_df.loc[row.name, max_col] = f'<span class=\"pink-love\">{bands_df.loc[row.name, max_col]}</span>'
                elif max_col in ['Good (70-84)', 'Optimal (85-100)']:
                    bands_df.loc[row.name, max_col] = f'<span class=\"good\">{bands_df.loc[row.name, max_col]}</span>'

    # Preventive comparison table (general healthy standards + your data snapshot)
    sleep_mean = sp_sleep_hours.get('mean')
    hr_mean = sp_nightly_hr.get('mean')
    spo2_mean = sp_spo2.get('mean')
    temp_mean = sp_temp_dev.get('mean')
    sleep_score_mean = sp_sleep_score.get('mean')
    readiness_mean = sp_readiness.get('mean')
    steps_mean = sp_steps.get('mean')

    def risk_sleep() -> str:
        if (sleep_mean is not None and sleep_mean < 6.5) or (not math.isnan(sleep_under7) and sleep_under7 >= 50):
            return 'red'
        if (sleep_mean is not None and sleep_mean < 7.0) or (not math.isnan(sleep_under7) and sleep_under7 >= 30):
            return 'yellow'
        return 'green'

    def risk_hr() -> str:
        if (hr_mean is not None and (hr_mean < 60 or hr_mean > 100)) or (not math.isnan(nightly_hr_out) and nightly_hr_out >= 20):
            return 'red'
        if not math.isnan(nightly_hr_out) and nightly_hr_out >= 10:
            return 'yellow'
        return 'green'

    def risk_spo2() -> str:
        if (spo2_mean is not None and spo2_mean < 94.0) or (not math.isnan(spo2_below95) and spo2_below95 >= 10):
            return 'red'
        if (spo2_mean is not None and spo2_mean < 95.0) or (not math.isnan(spo2_below95) and spo2_below95 >= 3):
            return 'yellow'
        return 'green'

    def risk_temp() -> str:
        if (temp_mean is not None and abs(temp_mean) > 0.5) or (not math.isnan(temp_dev_over) and temp_dev_over >= 35):
            return 'red'
        if (temp_mean is not None and abs(temp_mean) > 0.3) or (not math.isnan(temp_dev_over) and temp_dev_over >= 20):
            return 'yellow'
        return 'green'

    def risk_sleep_score() -> str:
        if sleep_score_mean is not None and sleep_score_mean < 60:
            return 'red'
        if sleep_score_mean is not None and sleep_score_mean < 70:
            return 'yellow'
        return 'green'

    def risk_readiness() -> str:
        if readiness_mean is not None and readiness_mean < 60:
            return 'red'
        if readiness_mean is not None and readiness_mean < 70:
            return 'yellow'
        return 'green'

    def risk_steps() -> str:
        low5000 = float((to_num(dailyactivity['steps']).dropna() < 5000).mean() * 100) if 'steps' in dailyactivity.columns and not to_num(dailyactivity['steps']).dropna().empty else math.nan
        if (steps_mean is not None and steps_mean < 4000) or (not math.isnan(low5000) and low5000 >= 60):
            return 'red'
        if (steps_mean is not None and steps_mean < 7000) or (not math.isnan(low5000) and low5000 >= 40):
            return 'yellow'
        return 'green'

    sleep_level = risk_sleep()
    hr_level = risk_hr()
    spo2_level = risk_spo2()
    temp_level = risk_temp()
    sleep_score_level = risk_sleep_score()
    readiness_level = risk_readiness()
    steps_level = risk_steps()

    def lvl_text(level: str) -> str:
        if level == 'red':
            return 'High / 高'
        if level == 'yellow':
            return 'Moderate / 中'
        return 'Low / 低'

    compare_rows = [
        {
            'Metric<br>指標': 'Main sleep duration<br>主要睡眠時數',
            'Risk level': risk_badge(sleep_level),
            'Unit': 'hours/night',
            'Your mean': color_text(sleep_level, fmt(sp_sleep_hours.get('mean'), 2)),
            'Your data spread': fmt_profile(sp_sleep_hours, 2),
            'Healthy population standard': 'Adults commonly need 7-9 h/night; minimum target >=7 h (CDC)',
            'Guideline / target': 'CDC sleep duration guidance',
            'Out-of-range rate': color_text(sleep_level, f"{sleep_under7:.1f}% nights < 7 h") if not math.isnan(sleep_under7) else 'N/A',
            'Easy interpretation': 'Most preventive benefit is when sleep is regularly at or above 7 hours.'
        },
        {
            'Metric<br>指標': 'Night average heart rate<br>夜間平均心率',
            'Risk level': risk_badge(hr_level),
            'Unit': 'bpm',
            'Your mean': color_text(hr_level, fmt(sp_nightly_hr.get('mean'), 1)),
            'Your data spread': fmt_profile(sp_nightly_hr, 1),
            'Healthy population standard': 'General resting HR range: 60-100 bpm (lower can be normal in fit adults)',
            'Guideline / target': 'AHA resting heart-rate reference',
            'Out-of-range rate': color_text(hr_level, f"{nightly_hr_out:.1f}% nights outside 60-100") if not math.isnan(nightly_hr_out) else 'N/A',
            'Easy interpretation': 'Your sleep HR is mostly in normal resting range; track trend changes, not one-day spikes.'
        },
        {
            'Metric<br>指標': 'Blood oxygen (SpO2)<br>血氧',
            'Risk level': risk_badge(spo2_level),
            'Unit': '%',
            'Your mean': color_text(spo2_level, fmt(sp_spo2.get('mean'), 2)),
            'Your data spread': fmt_profile(sp_spo2, 2),
            'Healthy population standard': 'Pulse oximetry normal range: about 95%-100%',
            'Guideline / target': 'MedlinePlus pulse-ox reference',
            'Out-of-range rate': color_text(spo2_level, f"{spo2_below95:.1f}% days < 95%") if not math.isnan(spo2_below95) else 'N/A',
            'Easy interpretation': 'Values are generally within the common normal pulse-ox range.'
        },
        {
            'Metric<br>指標': 'Temperature deviation<br>體溫偏差',
            'Risk level': risk_badge(temp_level),
            'Unit': 'degC deviation',
            'Your mean': color_text(temp_level, fmt(sp_temp_dev.get('mean'), 2)),
            'Your data spread': fmt_profile(sp_temp_dev, 2),
            'Healthy population standard': 'No single universal healthy range for wearable deviation metrics',
            'Guideline / target': 'Use trend + symptoms; sustained upward shifts may indicate physiological stress',
            'Out-of-range rate': color_text(temp_level, f"{temp_dev_over:.1f}% days with |deviation| > 0.5") if not math.isnan(temp_dev_over) else 'N/A',
            'Easy interpretation': 'This is personal-trend based, not a fixed clinical thermometer range.'
        },
        {
            'Metric<br>指標': 'Sleep score<br>睡眠分數',
            'Risk level': risk_badge(sleep_score_level),
            'Unit': '0-100',
            'Your mean': color_text(sleep_score_level, fmt(sp_sleep_score.get('mean'), 1)),
            'Your data spread': fmt_profile(sp_sleep_score, 1),
            'Healthy population standard': 'Consumer wellness score (not a clinical diagnostic scale)',
            'Guideline / target': 'Oura bands: 85-100 optimal; 70-84 good; 60-69 fair; <60 pay attention',
            'Out-of-range rate': color_text(sleep_score_level, f"{float((to_num(dailysleep['score']).dropna() < 70).mean()*100):.1f}% days <70") if 'score' in dailysleep.columns and not to_num(dailysleep['score']).dropna().empty else 'N/A',
            'Easy interpretation': 'Use sub-70 days as recovery warning days.'
        },
        {
            'Metric<br>指標': 'Readiness score<br>準備度分數',
            'Risk level': risk_badge(readiness_level),
            'Unit': '0-100',
            'Your mean': color_text(readiness_level, fmt(sp_readiness.get('mean'), 1)),
            'Your data spread': fmt_profile(sp_readiness, 1),
            'Healthy population standard': 'Consumer wellness score (not a clinical diagnostic scale)',
            'Guideline / target': 'Oura bands: 85-100 optimal; 70-84 good; 60-69 fair; <60 pay attention',
            'Out-of-range rate': color_text(readiness_level, f"{float((to_num(readiness['score']).dropna() < 70).mean()*100):.1f}% days <70") if 'score' in readiness.columns and not to_num(readiness['score']).dropna().empty else 'N/A',
            'Easy interpretation': 'If readiness is low, consider reducing training load and improving sleep regularity.'
        },
        {
            'Metric<br>指標': 'Daily steps<br>每日步數',
            'Risk level': risk_badge(steps_level),
            'Unit': 'steps/day',
            'Your mean': color_text(steps_level, fmt(sp_steps.get('mean'), 0)),
            'Your data spread': fmt_profile(sp_steps, 0),
            'Healthy population standard': 'No universal step cutoff; active adults often aim for a stable daily walking routine',
            'Guideline / target': 'Public-health target: >=150 min/week moderate activity (CDC)',
            'Out-of-range rate': color_text(steps_level, f"{float((to_num(dailyactivity['steps']).dropna() < 5000).mean()*100):.1f}% days <5,000") if 'steps' in dailyactivity.columns and not to_num(dailyactivity['steps']).dropna().empty else 'N/A',
            'Easy interpretation': 'More important than one cutoff: keep a stable weekly active pattern.'
        },
    ]
    compare_df = pd.DataFrame(compare_rows)

    # Evidence-based medical inference and action plan
    # Priority ranking rule:
    # priority_score = medical_importance(1-5) + feasibility(1-5) + risk_weight
    # risk_weight: High=3, Moderate=2, Low=1
    rec_rows: list[dict[str, Any]] = []
    risk_weight = {'High / 高': 3, 'Moderate / 中': 2, 'Low / 低': 1}

    if not math.isnan(sleep_under7):
        rec_rows.append({
            '_medical_importance': 5,
            '_feasibility': 4,
            'Domain / 面向': 'Sleep duration / 睡眠時數',
            'Risk': lvl_text(sleep_level),
            'What data suggests / 資料推測': f'{sleep_under7:.1f}% nights are below 7h.',
            'Medical implication / 醫學意義': 'Chronic short sleep is linked with higher cardiometabolic and mood-risk burden.',
            'Action (2-4 weeks) / 建議行動（2-4 週）': 'Set fixed sleep window; target >=7h on >=5 nights/week; avoid caffeine 8h before bedtime.',
            'How to track / 追蹤方式': 'Track weekly: nights >=7h, Sleep Score trend, daytime fatigue.'
        })

    if not nightly_hr.empty:
        rec_rows.append({
            '_medical_importance': 4,
            '_feasibility': 4,
            'Domain / 面向': 'Night HR / 夜間心率',
            'Risk': lvl_text(hr_level),
            'What data suggests / 資料推測': f'Median {fmt(sp_nightly_hr.get("median"),1)} bpm; out-of-range nights: {fmt(nightly_hr_out,1)}%.',
            'Medical implication / 醫學意義': 'Rising night HR over several days can reflect physiological stress, poor recovery, illness, alcohol effect, or overreaching.',
            'Action (2-4 weeks) / 建議行動（2-4 週）': 'On days with elevated night HR + low readiness, reduce training intensity by 20-40% and prioritize recovery sleep.',
            'How to track / 追蹤方式': 'Use 7-day moving average of night HR, not single-day spikes.'
        })

    if 'spo2_avg' in dailyspo2.columns and not dailyspo2['spo2_avg'].dropna().empty:
        rec_rows.append({
            '_medical_importance': 4,
            '_feasibility': 3,
            'Domain / 面向': 'SpO2 / 血氧',
            'Risk': lvl_text(spo2_level),
            'What data suggests / 資料推測': f'Mean {fmt(sp_spo2.get("mean"),2)}%; days <95%: {fmt(spo2_below95,1)}%.',
            'Medical implication / 醫學意義': 'Persistent low SpO2 patterns can warrant clinical review, especially if snoring, daytime sleepiness, or breathing symptoms exist.',
            'Action (2-4 weeks) / 建議行動（2-4 週）': 'If low readings cluster with symptoms, discuss sleep/breathing evaluation with a clinician.',
            'How to track / 追蹤方式': 'Track low-SpO2 clusters (>=3 low days in 2 weeks) and symptom diary.'
        })

    rec_rows.append({
        '_medical_importance': 4,
        '_feasibility': 5,
        'Domain / 面向': 'Activity pattern / 活動型態',
        'Risk': lvl_text(steps_level),
        'What data suggests / 資料推測': f'Mean steps {fmt(steps_mean,0)}; low-step days (<5000): {float((to_num(dailyactivity["steps"]).dropna() < 5000).mean()*100):.1f}%.' if 'steps' in dailyactivity.columns and not to_num(dailyactivity['steps']).dropna().empty else 'N/A',
        'Medical implication / 醫學意義': 'In preventive medicine, regular moderate activity improves cardiovascular and metabolic outcomes.',
        'Action (2-4 weeks) / 建議行動（2-4 週）': 'Build routine: >=150 min/week moderate activity; add short walking breaks every 60-90 min sedentary time.',
        'How to track / 追蹤方式': 'Track weekly active minutes + step consistency (variance week-to-week).'
    })

    rec_rows.append({
        '_medical_importance': 3,
        '_feasibility': 5,
        'Domain / 面向': 'Readiness + sleep score / 準備度與睡眠分數',
        'Risk': lvl_text('yellow' if (sleep_score_level != 'green' or readiness_level != 'green') else 'green'),
        'What data suggests / 資料推測': f'Sleep score mean {fmt(sleep_score_mean,1)}, readiness mean {fmt(readiness_mean,1)}.',
        'Medical implication / 醫學意義': 'When both recovery indicators drop together, injury risk and overtraining risk increase.',
        'Action (2-4 weeks) / 建議行動（2-4 週）': 'Use a simple rule: if both scores are low for >=2 days, deload and prioritize sleep, hydration, and lower-intensity training.',
        'How to track / 追蹤方式': 'Track 2-day and 7-day trends; avoid decisions from one low day only.'
    })

    for r in rec_rows:
        rw = risk_weight.get(r.get('Risk', 'Low / 低'), 1)
        r['_priority_score'] = int(r['_medical_importance']) + int(r['_feasibility']) + rw
    rec_rows = sorted(rec_rows, key=lambda x: x['_priority_score'], reverse=True)
    for i, r in enumerate(rec_rows, start=1):
        r['Priority'] = str(i)

    rec_df = pd.DataFrame(rec_rows)

    # Key preventive insights
    insight_lines = [
        f"Sleep duration median is {fmt(sp_sleep_hours.get('median'), 2)} h/night. Nights below 7h: {fmt(sleep_under7,1)}%.",
        f"Night heart-rate median is {fmt(sp_nightly_hr.get('median'),1)} bpm (P10-P90: {fmt(sp_nightly_hr.get('p10'),1)}-{fmt(sp_nightly_hr.get('p90'),1)}).",
        f"SpO2 median is {fmt(sp_spo2.get('median'),2)}% and below-95% days are {fmt(spo2_below95,1)}%.",
        f"Temperature deviation median is {fmt(sp_temp_dev.get('median'),2)} degC (interpret with multi-day trend + symptoms).",
        f"Sleep score median is {fmt(sp_sleep_score.get('median'),0)} and readiness median is {fmt(sp_readiness.get('median'),0)} (Oura score bands).",
    ]

    # Trend charts
    charts: list[tuple[str, str]] = []

    def add_chart(df: pd.DataFrame, xcol: str, ycol: str, label: str, title: str, color: str, file: str):
        if xcol in df.columns and ycol in df.columns:
            d = df[[xcol, ycol]].dropna().sort_values(xcol)
            if not d.empty:
                xs = [str(x)[:10] for x in d[xcol].tolist()]
                ys = [float(y) for y in d[ycol].tolist()]
                charts.append((label, save_chart(file, line_svg(xs, ys, title, color))))

    add_chart(dailyactivity, 'day_dt', 'steps', 'Daily Steps / 每日步數', 'Daily Steps Trend', '#ff6b8b', 'steps.svg')
    add_chart(dailysleep, 'day_dt', 'score', 'Sleep Score / 睡眠分數', 'Sleep Score Trend', '#5e60ce', 'sleep.svg')
    add_chart(readiness, 'day_dt', 'score', 'Readiness Score / 準備度分數', 'Readiness Score Trend', '#2a9d8f', 'readiness.svg')

    hr_daily = pd.DataFrame()
    if 'ts' in heartrate.columns and 'bpm' in heartrate.columns:
        hr_daily = heartrate.dropna(subset=['ts', 'bpm']).copy()
        hr_daily['day_dt'] = hr_daily['ts'].dt.tz_convert('UTC').dt.date
        hr_daily = hr_daily.groupby('day_dt', as_index=False)['bpm'].mean()
    add_chart(hr_daily, 'day_dt', 'bpm', 'Heart Rate (Daily Avg) / 心率（日均）', 'Daily Average Heart Rate', '#e63946', 'heart.svg')

    temp_daily = pd.DataFrame()
    if 'ts' in temperature.columns and 'skin_temp' in temperature.columns:
        temp_daily = temperature.dropna(subset=['ts', 'skin_temp']).copy()
        temp_daily['day_dt'] = temp_daily['ts'].dt.tz_convert('UTC').dt.date
        temp_daily = temp_daily.groupby('day_dt', as_index=False)['skin_temp'].mean()
    add_chart(temp_daily, 'day_dt', 'skin_temp', 'Skin Temperature (Daily Avg) / 皮膚溫度（日均）', 'Daily Average Skin Temperature', '#f4a261', 'temp.svg')

    if 'ts' in daystress.columns:
        ds = daystress.dropna(subset=['ts']).copy()
        ds['day_dt'] = ds['ts'].dt.tz_convert('UTC').dt.date
        ds = ds.groupby('day_dt', as_index=False)[['stress_value', 'recovery_value']].mean()
        if not ds.empty:
            xs = [str(x)[:10] for x in ds['day_dt'].tolist()]
            charts.append(('Stress vs Recovery / 壓力與恢復', save_chart('stress_recovery.svg', dual_line_svg(xs, ds['stress_value'].tolist(), ds['recovery_value'].tolist(), 'Daily Stress vs Recovery'))))

    # Utility for html table
    def table_html(df: pd.DataFrame, max_rows: int = 300) -> str:
        if df is None or df.empty:
            return '<p class="muted">No data available.</p>'
        return df.head(max_rows).to_html(index=False, classes='nice-table', border=0, escape=False)

    def rec_cards_html(items: list[dict[str, str]]) -> str:
        if not items:
            return '<p class="muted">No recommendation data available.</p>'
        cards = []
        for it in items:
            risk_text = it.get('Risk', 'Low / 低')
            risk_class = 'rec-risk-alert' if ('Moderate' in risk_text or 'High' in risk_text) else 'rec-risk-low'
            action_class = 'rec-action-alert' if ('Moderate' in risk_text or 'High' in risk_text) else ''
            cards.append(
                f"""
<article class="rec-card">
  <div class="rec-head">
    <span class="rec-priority">Priority {it.get('Priority','-')}</span>
    <span class="rec-risk {risk_class}">{risk_text}</span>
  </div>
  <h3>{it.get('Domain / 面向','')}</h3>
  <p><b>Data Suggests / 資料推測：</b> {it.get('What data suggests / 資料推測','')}</p>
  <p><b>Medical Meaning / 醫學意義：</b> {it.get('Medical implication / 醫學意義','')}</p>
  <p><b>Action 2-4 Weeks / 行動建議：</b> <span class="{action_class}">{it.get('Action (2-4 weeks) / 建議行動（2-4 週）','')}</span></p>
  <p><b>Track / 追蹤：</b> {it.get('How to track / 追蹤方式','')}</p>
</article>
""".strip()
            )
        return '<div class="rec-grid">' + ''.join(cards) + '</div>'

    sources_html = """
<ul>
  <li><a href=\"https://www.heart.org/en/healthy-living/fitness/fitness-basics/target-heart-rates\" target=\"_blank\">American Heart Association: Resting heart rate</a></li>
  <li><a href=\"https://www.cdc.gov/sleep/about/index.html\" target=\"_blank\">CDC: Recommended sleep duration</a></li>
  <li><a href=\"https://medlineplus.gov/lab-tests/pulse-oximetry\" target=\"_blank\">MedlinePlus: Pulse oximetry normal range</a></li>
  <li><a href=\"https://support.ouraring.com/hc/en-us/articles/360025445574-An-Introduction-to-Your-Sleep-Score\" target=\"_blank\">Oura: Score bands</a></li>
  <li><a href=\"https://support.ouraring.com/hc/en-us/articles/360025587493\" target=\"_blank\">Oura: Temperature baseline/deviation interpretation</a></li>
</ul>
"""

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Oura Preventive Health Report</title>
  <style>
    :root {{ --bg1:#fff7f9; --bg2:#eef9ff; --card:#fff; --ink:#273043; --muted:#5f6c7b; --line:#e8ecf1; --accent:#ff6b8b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:'Avenir Next','PingFang TC','Microsoft JhengHei',sans-serif; color:var(--ink); background:radial-gradient(circle at top left,var(--bg1),#fff 45%),radial-gradient(circle at top right,var(--bg2),#fff 35%); line-height:1.6; }}
    .container {{ max-width:1180px; margin:0 auto; padding:24px; }}
    .hero {{ background:linear-gradient(135deg,#ffe4ec,#e7f8ff); border:1px solid var(--line); border-radius:20px; padding:26px; box-shadow:0 10px 26px rgba(30,43,67,.08); }}
    h1 {{ margin:0 0 8px; font-size:30px; }} h2 {{ margin:26px 0 12px; font-size:22px; }}
    .subtitle {{ color:var(--muted); }}
    .card {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 6px 20px rgba(30,43,67,.06); }}
    .grid {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); }}
    .pill {{ display:inline-block; padding:6px 10px; border-radius:999px; background:#fff; border:1px solid var(--line); margin-right:6px; font-size:12px; }}
    .nice-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    .nice-table th,.nice-table td {{ border:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }}
    .nice-table th {{ background:#f6f9fc; }}
    .chart-grid {{ display:grid; gap:14px; grid-template-columns:repeat(2, minmax(0, 1fr)); }}
    .chart-item svg {{ width:100%; height:auto; border:1px solid var(--line); border-radius:12px; }}
    .muted {{ color:var(--muted); }}
    .footnote-sm {{ font-size:12px; line-height:1.45; color:#4f5d70; margin-top:8px; }}
    .footnote-sm ul {{ margin:6px 0 6px 18px; padding:0; }}
    .footnote-sm p {{ margin:6px 0 0; }}
    .danger {{ color:#c1121f; font-weight:700; }}
    .warn {{ color:#b26a00; font-weight:700; }}
    .good {{ color:#2a9d50; font-weight:700; }}
    .badge {{ display:inline-block; padding:3px 8px; border-radius:999px; font-size:12px; font-weight:700; border:1px solid currentColor; white-space:nowrap; }}
    .pink-love {{ color:#ff4f87; font-weight:700; }}
    .rec-grid {{ display:grid; grid-template-columns:1fr; gap:12px; }}
    .rec-card {{ border:1px solid var(--line); border-radius:12px; padding:12px; background:#fff; }}
    .rec-card h3 {{ margin:6px 0 8px; font-size:16px; }}
    .rec-card p {{ margin:6px 0; font-size:13px; }}
    .rec-head {{ display:flex; justify-content:space-between; align-items:center; gap:8px; }}
    .rec-priority {{ font-size:12px; padding:2px 8px; border-radius:999px; background:#e6f6ec; color:#15693f; border:1px solid #9ad6b3; font-weight:700; }}
    .rec-risk {{ font-size:12px; padding:2px 8px; border-radius:999px; font-weight:700; border:1px solid transparent; }}
    .rec-risk-alert {{ background:#ffe3ef; color:#8a1f4d; border-color:#ffb9d3; }}
    .rec-risk-low {{ background:#e8f7ed; color:#166b40; border-color:#a8dfbc; }}
    .rec-action-alert {{ color:#b0175f; font-weight:700; }}
    @media (max-width: 900px) {{
      .chart-grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <section class=\"hero\">
      <h1>Preventive Health Deep-Dive / 預防醫學深度報告</h1>
      <div class=\"subtitle\">Your Oura data with healthy-population standards, practical ranges, and easy interpretation.</div>
      <div style=\"margin-top:10px\"><span class=\"pill\">Data window: {window_text}</span><span class=\"pill\">Approach: long-term trends + healthy-population standards + your own data distribution</span></div>
    </section>

    <h2>1) Analysis Approach / 分析方法</h2>
    <div class=\"card\">
      <ul>
        <li>Use general healthy standards first, then compare with your own data spread (median and P10-P90).</li>
        <li>Sleep, readiness, HR, SpO2 and temperature deviation are useful for early preventive signals.</li>
        <li>Single-day spikes are less important than multi-day trends (especially temperature and stress metrics).</li>
      </ul>
    </div>

    <h2>2) Preventive Comparison Table / 預防醫學比較表（含健康族群標準）</h2>
    <div class=\"card\">{table_html(compare_df)}</div>
    <p class=\"muted footnote-sm\">Color rule: <span class=\"danger\">Red</span> = high risk, <span class=\"warn\">Yellow</span> = mild caution, <span class=\"good\">Green</span> = relatively stable.</p>

    <h2>3) Oura Score Band Distribution / 分數區間分佈</h2>
    <div class=\"card\">{table_html(bands_df)}
      <div class=\"footnote-sm\">
        <p><b>Footnote / 註解</b></p>
        <ul>
          <li><b>Activity Score / 活動分數</b>: Reflects daily movement and activity balance (e.g., steps, active time, inactivity, training load).<br/>代表每日活動表現與平衡（如步數、活動時間、久坐狀態、運動負荷）。</li>
          <li><b>Sleep Score / 睡眠分數</b>: Reflects overnight sleep quality and structure (e.g., total sleep, efficiency, timing, deep/REM, restfulness).<br/>代表夜間睡眠品質與結構（如總睡眠、效率、作息時機、深睡/REM、安穩度）。</li>
          <li><b>Readiness Score / 準備度分數</b>: Reflects recovery readiness for the day (e.g., HR/HRV signals, recent sleep and activity strain).<br/>代表當日恢復與可負荷程度（如心率/HRV 訊號、近期睡眠與活動壓力）。</li>
        </ul>
        <p><b>How this is evaluated / 怎麼評估：</b> Oura 會把多個生理與行為訊號整合成 0-100 分。分數越高通常代表狀態越穩定；長期趨勢比單日波動更有判讀價值。</p>
      </div>
    </div>

    <h2>4) Data-Driven Clinical Notes / 醫學解讀筆記</h2>
    <div class=\"card\">
      <ul>{''.join([f'<li>{x}</li>' for x in insight_lines])}</ul>
      <ul>
        <li>這份比較先用健康族群的一般參考值，搭配你的數據表現，幫你快速看出目前最需要優先處理的項目。</li>
        <li>你現在最值得投資的第一步，是把睡眠時數與規律性先穩住；若長期低於 7 小時，恢復效率通常會明顯下滑。</li>
        <li>夜間平均心率請看「連續 3-7 天趨勢」：若持續偏高，再配合壓力、疲勞與活動量一起判讀會更準確。</li>
        <li>血氧多數人常見在 95%-100%；若你出現連續偏低，建議同時觀察是否有打鼾、白天嗜睡或呼吸不適，再考慮就醫評估。</li>
        <li>體溫偏差屬於趨勢型訊號，不是單一診斷值；重點是「是否連續異常」與「是否合併症狀」。</li>
      </ul>
    </div>

    <h2>5) Trend Graphs / 趨勢圖</h2>
    <div class=\"chart-grid\">{''.join([f'<div class="chart-item card"><h3>{t}</h3>{(ASSET_DIR/f).read_text(encoding="utf-8")}</div>' for t, f in charts])}</div>

    <h2>6) Medical Inference, Prediction, and Action Plan / 醫學推測、預測與行動建議</h2>
    <div class=\"card\">{rec_cards_html(rec_rows)}</div>
    <div class=\"card\" style=\"margin-top:10px;\">
      <p><b>Summary / 總結</b></p>
      <p>
        You are already doing many things well, and your data gives a clear path forward.<br/>
        The most valuable next step is to protect sleep consistency first.<br/>
        Then support recovery with steady daytime activity and lighter training on high-stress or low-readiness days.
      </p>
      <p>
        你其實已經做得不錯了，現在的數據也很清楚地指出下一步方向。<br/>
        最值得優先做的是先把睡眠規律穩住，<br/>
        再用穩定的日間活動量與適度調整訓練負荷來支持恢復。
      </p>
    </div>
    <p class=\"muted footnote-sm\">Priority is ranked by: medical importance + feasibility + risk level (High/Moderate/Low). This section is decision-support, not a diagnosis.</p>

    <h2>7) References / 參考來源</h2>
    <div class=\"card\">{sources_html}</div>

    <p class=\"muted footnote-sm\">Medical disclaimer: This report is for health education and early screening support only. It is not a medical diagnosis, treatment plan, or emergency assessment. If you have persistent symptoms, abnormal readings, or urgent concerns, please seek care from a qualified clinician promptly. The safest way to use this report is to combine your data trends with symptoms, medical history, and professional advice.</p>
  </div>
</body>
</html>
"""

    out_name = Path(args.output_html)
    out = out_name if out_name.is_absolute() else (BASE / out_name)
    out.write_text(html, encoding='utf-8')
    print(f'Updated {out}')


if __name__ == '__main__':
    main()
