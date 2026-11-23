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


def dual_line_svg(xs: list[str], y1: list[float], y2: list[float], title: str, label1: str = 'Stress', label2: str = 'Recovery') -> str:
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
  <rect x=\"{margin + chart_w - 170}\" y=\"{margin + 8}\" width=\"10\" height=\"10\" fill=\"#ff7f50\"/><text x=\"{margin + chart_w - 155}\" y=\"{margin + 17}\" font-size=\"11\" fill=\"#5f6c7b\">{label1}</text>
  <rect x=\"{margin + chart_w - 90}\" y=\"{margin + 8}\" width=\"10\" height=\"10\" fill=\"#52b788\"/><text x=\"{margin + chart_w - 75}\" y=\"{margin + 17}\" font-size=\"11\" fill=\"#5f6c7b\">{label2}</text>
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



# ---------------------------------------------------------------------------
# Estimated Resilience (added 2026-11): a transparent, documented estimate
# inspired by the Oura app's Resilience feature. It is NOT Oura's proprietary
# score. Method (v3): contributors over 14-day windows - nighttime recovery =
# 70% sleep component (sleep score minus 8 points per hour below 7 h) + 30%
# physiology component (HRV and lowest resting HR vs the wearer's own
# baseline-month medians); daytime recovery from restorative minutes;
# daytime stress load from stress minutes. The level index blends the 14-day
# value (40%) with a 28-day value (60%). Bands: Limited <40, Adequate 40-54,
# Solid 55-64, Strong 65-79, Exceptional >=80.
# ---------------------------------------------------------------------------

_RES_LEVELS = [('Exceptional', 80, '#2f9e44'), ('Strong', 65, '#12a594'),
               ('Solid', 55, '#d9930d'), ('Adequate', 40, '#e8590c'),
               ('Limited', 0, '#e03131')]


def _res_level(v: float) -> str:
    for name, lo, _c in _RES_LEVELS:
        if v >= lo:
            return name
    return 'Limited'


def _res_band_rec(v: float) -> str:
    return 'Optimal' if v >= 80 else 'Good' if v >= 65 else 'Fair' if v >= 50 else 'Pay attention'


def _res_band_stress(v: float) -> str:
    return 'Low' if v < 33 else 'Moderate' if v < 66 else 'High'


def compute_resilience(dailystress: pd.DataFrame, dailysleep: pd.DataFrame,
                       sleepmodel: pd.DataFrame) -> dict | None:
    """Return the v3 estimate dict, or None when history is insufficient."""
    if dailystress.empty or 'day' not in dailystress.columns:
        return None
    st = dailystress.copy()
    st['day'] = pd.to_datetime(st['day'], errors='coerce')
    for c in ['stress_high', 'recovery_high']:
        if c not in st.columns:
            return None
        st[c] = to_num(st[c])
    st['stress_min'] = st['stress_high'] / 60.0
    st['rec_min'] = st['recovery_high'] / 60.0
    if 'day_summary' not in st.columns:
        st['day_summary'] = ''
    st = st.dropna(subset=['day']).sort_values('day')

    sl = dailysleep.copy()
    if not sl.empty and 'day' in sl.columns and 'score' in sl.columns:
        sl['day'] = pd.to_datetime(sl['day'], errors='coerce')
        sl['score'] = to_num(sl['score'])
        st = st.merge(sl[['day', 'score']].rename(columns={'score': 'sleep_score'}),
                      on='day', how='left')
    else:
        st['sleep_score'] = pd.NA

    sm = sleepmodel.copy()
    if not sm.empty and 'day' in sm.columns and 'total_sleep_duration' in sm.columns:
        sm['day'] = pd.to_datetime(sm['day'], errors='coerce')
        for c in ['total_sleep_duration', 'average_hrv', 'lowest_heart_rate']:
            if c in sm.columns:
                sm[c] = to_num(sm[c])
        sm = sm.sort_values('total_sleep_duration').groupby('day').tail(1)
        keep = ['day', 'total_sleep_duration'] + [c for c in ['average_hrv', 'lowest_heart_rate'] if c in sm.columns]
        st = st.merge(sm[keep], on='day', how='left')
    else:
        st['total_sleep_duration'] = pd.NA
    st['sleep_hrs'] = st['total_sleep_duration'] / 3600.0

    if len(st) < 7:
        return None
    limited_history = len(st) < 28

    # Baseline month: the first 31 days when the dataset reaches >=45 days
    # back; otherwise the earliest days before the final 28 (>=7 required).
    span_days = (st['day'].max() - st['day'].min()).days
    if span_days >= 45:
        base = st[st['day'] <= st['day'].min() + pd.Timedelta(days=30)]
    elif len(st) >= 35:
        base = st.iloc[:-28]
    else:
        base = st.iloc[:max(3, len(st) // 4)]
    if len(base) < 3 or 'average_hrv' not in st.columns or 'lowest_heart_rate' not in st.columns:
        return None
    hrv_base = float(base['average_hrv'].median())
    rhr_base = float(base['lowest_heart_rate'].median())
    if pd.isna(hrv_base) or pd.isna(rhr_base):
        return None

    def _night_sleep(r):
        if pd.isna(r['sleep_score']):
            return float('nan')
        hrs = r['sleep_hrs'] if not pd.isna(r['sleep_hrs']) else 7.0
        return float(r['sleep_score']) - max(0.0, 7.0 - hrs) * 8.0

    st['ns'] = st.apply(_night_sleep, axis=1)
    st['phys'] = (50 + (st['average_hrv'] - hrv_base) * 1.5
                  - (st['lowest_heart_rate'] - rhr_base) * 3.0).clip(0, 100)
    st['night'] = 0.7 * st['ns'] + 0.3 * st['phys']
    st['rec_score'] = (st['rec_min'] / 120 * 100).clip(0, 100)
    st['stress_score'] = (st['stress_min'] / 300 * 100).clip(0, 100)
    mp14 = 12 if len(st) >= 14 else max(5, len(st) - 2)
    for c in ['night', 'rec_score', 'stress_score']:
        st[c + '_14'] = st[c].rolling(14, min_periods=mp14).mean()
        st[c + '_28'] = st[c].rolling(28, min_periods=24).mean()
    st['idx14'] = 0.4 * st.night_14 + 0.3 * st.rec_score_14 + 0.3 * (100 - st.stress_score_14)
    st['idx28'] = 0.4 * st.night_28 + 0.3 * st.rec_score_28 + 0.3 * (100 - st.stress_score_28)
    if limited_history:
        st['index'] = st['idx14']
    else:
        st['index'] = 0.4 * st.idx14 + 0.6 * st.idx28
    st['stress_r'] = st['stress_min'].rolling(14, min_periods=12).mean()
    st['rec_r'] = st['rec_min'].rolling(14, min_periods=12).mean()

    win = st.tail(14)
    last = st.dropna(subset=['index']).iloc[-1]
    night_v = float(win['night'].mean())
    rec_v = float(win['rec_score'].mean())
    smin = float(win['stress_min'].mean())
    return {
        'index': round(float(last['index']), 1),
        'level': _res_level(float(last['index'])),
        'window': (str(win['day'].min().date()), str(win['day'].max().date())),
        'contributors': {
            'nighttime_recovery': (round(night_v, 1), _res_band_rec(night_v)),
            'daytime_recovery': (round(rec_v, 1), _res_band_rec(rec_v), int(round(float(win['rec_min'].mean())))),
            'daytime_stress_load': (round(smin / 300 * 100, 1), _res_band_stress(smin / 300 * 100), int(round(smin))),
        },
        'trend': [(str(r['day'].date()), round(float(r['index']), 1))
                  for _, r in st.dropna(subset=['index']).tail(14).iterrows()],
        'map': [(str(r['day'].date()), round(float(r['stress_r']), 1), round(float(r['rec_r']), 1))
                for _, r in win.iterrows() if not pd.isna(r['stress_r'])],
        'stress_days': [(str(r['day'].date()), float(r['stress_min']), float(r['rec_min']),
                         str(r.get('day_summary') or '')) for _, r in win.iterrows()],
        'daily': [(str(r['day'].date()),
                   None if pd.isna(r['index']) else round(float(r['index']), 1),
                   None if pd.isna(r['stress_r']) else round(float(r['stress_r']), 1),
                   None if pd.isna(r['rec_r']) else round(float(r['rec_r']), 1),
                   float(r['stress_min']), float(r['rec_min']),
                   str(r.get('day_summary') or '')) for _, r in st.iterrows()],
        'limited_history': limited_history,
        'hrv_base': hrv_base, 'rhr_base': rhr_base,
    }


def resilience_trend_svg(res: dict) -> str:
    w, h, pad_l, pad_b, pad_t = 560, 240, 8, 22, 10
    plot_w, plot_h = w - pad_l - 70, h - pad_b - pad_t
    bands = [('Exceptional', 80, 100, '#e6f4ea'), ('Strong', 65, 80, '#e0f2f1'),
             ('Solid', 55, 65, '#fdf3d7'), ('Adequate', 40, 55, '#fdebd7'),
             ('Limited', 0, 40, '#fbe3e3')]
    out = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    for name, lo, hi, color in bands:
        y = pad_t + (100 - hi) / 100 * plot_h
        bh = (hi - lo) / 100 * plot_h
        out.append(f'<rect x="{pad_l}" y="{y:.0f}" width="{plot_w}" height="{bh:.0f}" fill="{color}"/>')
        out.append(f'<text x="{pad_l + plot_w + 6}" y="{y + bh / 2 + 4:.0f}" font-size="11" fill="#555">{name}</text>')
    pts = []
    n = max(len(res['trend']) - 1, 1)
    for i, (d, v) in enumerate(res['trend']):
        x = pad_l + i / n * plot_w
        y = pad_t + (100 - v) / 100 * plot_h
        pts.append(f'{x:.0f},{y:.0f}')
        out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3" fill="#333"/>')
        if i in (0, len(res['trend']) - 1):
            out.append(f'<text x="{x:.0f}" y="{h - 6}" font-size="10" fill="#555" text-anchor="middle">{d[5:]}</text>')
    out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#333" stroke-width="1.6"/>')
    out.append('</svg>')
    return ''.join(out)


def resilience_map_svg(res: dict) -> str:
    w, h, pad = 560, 330, 30
    plot_w, plot_h = w - 2 * pad, h - 2 * pad
    left = [0.05, 0.20, 0.35, 0.50]
    ctrl = [0.15, 0.30, 0.45, 0.60]
    right = [0.45, 0.60, 0.75, 0.90]
    colors = ['#2f9e44', '#12a594', '#d9930d', '#e8590c', '#e03131']
    names = ['Exceptional', 'Strong', 'Solid', 'Adequate', 'Limited']
    out = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<line x1="{pad + plot_w / 2}" y1="{pad}" x2="{pad + plot_w / 2}" y2="{pad + plot_h}" stroke="#bbb" stroke-width="1"/>')
    out.append(f'<line x1="{pad}" y1="{pad + 0.38 * plot_h}" x2="{pad + plot_w}" y2="{pad + 0.38 * plot_h}" stroke="#bbb" stroke-width="1"/>')
    for i in range(4):
        x0, y0 = pad, pad + left[i] * plot_h
        cx, cy = pad + plot_w / 2, pad + ctrl[i] * plot_h
        x1, y1 = pad + plot_w, pad + right[i] * plot_h
        out.append(f'<path d="M {x0} {y0:.0f} Q {cx} {cy:.0f} {x1} {y1:.0f}" fill="none" stroke="#888" stroke-width="1.4" stroke-dasharray="5 4"/>')
    for i, (name, color) in enumerate(zip(names, colors)):
        mid = (left[i - 1] + ctrl[i - 1]) / 2 if i > 0 else 0.02
        y = pad + (mid + 0.055) * plot_h if i > 0 else pad + 0.09 * plot_h
        if i == 4:
            y = pad + 0.86 * plot_h
        out.append(f'<text x="{pad + 0.14 * plot_w}" y="{y:.0f}" font-size="11" font-weight="600" fill="{color}">{name}</text>')
    out.append(f'<text x="{w / 2}" y="16" font-size="11" fill="#555" text-anchor="middle">High recovery</text>')
    out.append(f'<text x="{w / 2}" y="{h - 6}" font-size="11" fill="#555" text-anchor="middle">Low recovery</text>')
    out.append(f'<text x="{pad}" y="{pad + 0.38 * plot_h - 6}" font-size="11" fill="#555">Low stress</text>')
    out.append(f'<text x="{pad + plot_w}" y="{pad + 0.38 * plot_h - 6}" font-size="11" fill="#555" text-anchor="end">High stress</text>')
    pts = []
    for i, (d, s, r) in enumerate(res['map']):
        x = pad + min(max(s, 0), 360) / 360 * plot_w
        y = pad + (1 - min(max(r + 48, 0), 240) / 240) * plot_h
        pts.append((x, y, d))
    for i in range(len(pts) - 1):
        out.append(f'<line x1="{pts[i][0]:.0f}" y1="{pts[i][1]:.0f}" x2="{pts[i + 1][0]:.0f}" y2="{pts[i + 1][1]:.0f}" stroke="#666" stroke-width="1.4"/>')
    for i, (x, y, d) in enumerate(pts):
        if i == len(pts) - 1:
            out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="7" fill="#ff4f87"/>')
            out.append(f'<text x="{x:.0f}" y="{y - 12:.0f}" font-size="10" fill="#ff4f87" text-anchor="middle">{d[5:]}</text>')
        else:
            out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3.4" fill="#fff" stroke="#666" stroke-width="1.4"/>')
            if i == 0:
                out.append(f'<text x="{x:.0f}" y="{y - 10:.0f}" font-size="10" fill="#555" text-anchor="middle">{d[5:]}</text>')
    out.append('</svg>')
    return ''.join(out)


def resilience_section(dailystress: pd.DataFrame, dailysleep: pd.DataFrame,
                       sleepmodel: pd.DataFrame) -> str:
    res = None
    try:
        res = compute_resilience(dailystress, dailysleep, sleepmodel)
    except Exception:
        res = None
    head = '<h2>6) Estimated Resilience / 估算韌性</h2>'
    if res is None:
        return (head + '<div class="card"><p>This section needs at least 7 days of '
                'daily stress, sleep, and sleep-session (HRV / resting HR) data. '
                '本節需要至少 7 天的每日壓力、睡眠與睡眠時段（HRV／靜息心率）資料。</p></div>')
    c = res['contributors']
    rows = ''.join([
        f"<tr><td>Nighttime recovery 夜間恢復</td><td>{c['nighttime_recovery'][0]}</td><td>{c['nighttime_recovery'][1]}</td></tr>",
        f"<tr><td>Daytime recovery 日間恢復</td><td>{c['daytime_recovery'][0]}</td><td>{c['daytime_recovery'][1]} (avg {c['daytime_recovery'][2]} min/day)</td></tr>",
        f"<tr><td>Daytime stress load 日間壓力負荷</td><td>{c['daytime_stress_load'][0]}</td><td>{c['daytime_stress_load'][1]} (avg {c['daytime_stress_load'][2]} min/day)</td></tr>",
    ])
    def _bars(days):
        if not days:
            return '<p class="muted">No data.</p>'
        mx = max(max(d[1], d[2]) for d in days) or 1
        out = []
        for d, s, r, label in days:
            chip = f' <span class="badge">{label}</span>' if label else ''
            out.append(
                f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;font-size:12px">'
                f'<span style="width:44px">{d[5:]}</span>'
                f'<span style="display:inline-block;height:9px;background:#e8590c;width:{s / mx * 150:.0f}px"></span>'
                f'<span style="width:86px">stress {s:.0f}m</span>'
                f'<span style="display:inline-block;height:9px;background:#2f9e44;width:{r / mx * 150:.0f}px"></span>'
                f'<span>recovery {r:.0f}m</span>{chip}</div>')
        return ''.join(out)

    daily = res['daily']
    weeks = [daily[i:i + 7] for i in range(0, len(daily), 7)]
    if len(weeks) >= 2:
        default_days = weeks[-2] + weeks[-1]
    else:
        default_days = daily
    period_defs = [('Latest — past week + current period 最新（上週＋本期）', default_days)]
    for i, wk in enumerate(reversed(weeks)):
        period_defs.append((f'Week {len(weeks) - i} · {wk[0][0]} to {wk[-1][0]}', wk))
    period_html = []
    options = []
    for pi, (label, days) in enumerate(period_defs):
        sub = {'trend': [(d[0], d[1]) for d in days if d[1] is not None],
               'map': [(d[0], d[2], d[3]) for d in days if d[2] is not None]}
        sdays = [(d[0], d[4], d[5], d[6]) for d in days]
        ts = sum(d[1] for d in sdays)
        tr = sum(d[2] for d in sdays)
        disp = 'block' if pi == 0 else 'none'
        period_html.append(
            f'<div class="res-period" id="resperiod-{pi}" style="display:{disp}">'
            f'<h3>Trend / 趨勢</h3>{resilience_trend_svg(sub) if sub["trend"] else "<p class=\"muted\">Not enough history yet for an index in this period. 本期歷史資料尚不足以計算指數。</p>"}'
            f'<h3>Map / 分佈圖</h3>{resilience_map_svg(sub) if sub["map"] else "<p class=\"muted\">Not enough history yet for the map in this period. 本期歷史資料尚不足以繪製地圖。</p>"}'
            f'<h3>Stress &amp; Recovery / 壓力與恢復</h3>'
            f'<p>Stress total {ts / 60:.1f} h · Recovery total {tr / 60:.1f} h · Daily average stress {ts / len(sdays):.0f} min, recovery {tr / len(sdays):.0f} min</p>'
            f'{_bars(sdays)}</div>')
        options.append(f'<option value="resperiod-{pi}">{label}</option>')
    onchange = ("(document.querySelectorAll('.res-period').forEach(p=>p.style.display='none'),"
                "document.getElementById(this.value).style.display='block')")
    periods_block = (f'<p><label>Period 期間：<select onchange="{onchange}">'
                     + ''.join(options) + '</select></label></p>' + ''.join(period_html))
    limited_note = ('<p class="warn">Limited history: with fewer than 28 days the estimate uses the 14-day window only and a shorter baseline — treat it as provisional. 歷史資料少於 28 天，指數僅採 14 日窗口與較短基線，請視為暫定值。</p>' if res.get('limited_history') else '')
    tot_s = sum(d[1] for d in res['stress_days'])
    tot_r = sum(d[2] for d in res['stress_days'])
    bars = _bars(res['stress_days'])
    return f"""{head}
    <div class="card">
      <p><b>Why resilience? 為何以韌性為核心？</b> Health resilience is the body's capacity to respond to stress, recover equilibrium, and repair damage — health is not merely the absence of disease, but the presence of this capacity. This report is organized around measuring it, detecting its decline, and supporting its repair. 健康韌性是身體回應壓力、恢復平衡、修復損傷的能力；健康不只是沒有疾病，而是擁有這樣的能力。本報告即圍繞「測量韌性、察覺衰退、支持修復」而設計。(Framework: Lu MC, keynote on Health Resilience — see References.)</p>
      <p style="font-size:18px"><b>Estimated level / 估算等級：{res['level']}</b> (index {res['index']}, window {res['window'][0]} to {res['window'][1]})</p>
      <p class="muted">A transparent estimate computed by this tool — not Oura's official Resilience score. 本工具之透明估算，非 Oura 官方韌性分數。</p>
      {limited_note}
      <table><tr><th>Contributor 貢獻因子</th><th>Score</th><th>Rating 評級</th></tr>{rows}</table>
      {periods_block}
      <p class="muted footnote-sm">Map points are 14-day rolling averages on fixed scales (stress 0-360 min, recovery 0-240 min); the pink dot is the most recent day. A fixed display offset is applied to the recovery axis so the trajectory registers against the level bands, as on Oura's map (whose recovery axis is normalized rather than raw minutes). The four dashed curves are the boundaries between the five Resilience levels. 地圖各點為 14 日滾動平均，採固定尺度；粉紅點為最近一日；恢復軸加上固定顯示偏移以對應等級曲線；四條虛線為五個韌性等級之分界。</p>
      <p class="muted footnote-sm">Method v3: nighttime recovery = 70% sleep component (sleep score - 8 points per hour below 7 h) + 30% physiology (50 + 1.5 per ms HRV above your baseline-month median of {res['hrv_base']:.0f} ms - 3 per bpm lowest resting HR above {res['rhr_base']:.0f} bpm, clipped 0-100). Daytime recovery score = avg recovery min / 120 x 100; stress load score = avg stress min / 300 x 100. Index = 0.4 x nighttime + 0.3 x daytime recovery + 0.3 x (100 - stress load), blending 14-day (40%) and 28-day (60%) windows. Levels: Limited &lt;40, Adequate 40-54, Solid 55-64, Strong 65-79, Exceptional &gt;=80.</p>
    </div>"""




def _score_band(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'N/A'
    return 'Optimal' if v >= 85 else 'Good' if v >= 70 else 'Fair' if v >= 60 else 'Pay attention'


def _ring_svg(value, label: str) -> str:
    r = 34
    circ = 2 * math.pi * r
    if value is None or (isinstance(value, float) and math.isnan(value)):
        frac, text = 0.0, 'N/A'
    else:
        frac, text = max(0.0, min(100.0, float(value))) / 100.0, f'{float(value):.0f}'
    return (
        f'<div style="text-align:center">'
        f'<svg width="92" height="92" viewBox="0 0 92 92">'
        f'<circle cx="46" cy="46" r="{r}" fill="none" stroke="#e5e5e5" stroke-width="8"/>'
        f'<circle cx="46" cy="46" r="{r}" fill="none" stroke="#333" stroke-width="8" '
        f'stroke-linecap="round" stroke-dasharray="{frac * circ:.0f} {circ:.0f}" '
        f'transform="rotate(-90 46 46)"/>'
        f'<text x="46" y="52" font-size="20" font-weight="700" text-anchor="middle" fill="#333">{text}</text>'
        f'</svg>'
        f'<div style="font-size:13px">{label}<br/><span class="muted">{_score_band(value)}</span></div></div>')


def summary_section(sleep_mean, readiness_mean, vitals, attention, stress14,
                    top_actions, window_text: str, n_days: int = 14) -> str:
    """Summary tab: score rings, vitals with risk dots, needs-attention,
    stress & recovery totals, and the top two actions (mirrors the app)."""
    lvl_color = {'green': '#2f9e44', 'yellow': '#d9930d', 'red': '#e03131'}
    vitals_html = ''.join(
        f'<span style="display:inline-block;margin:3px 10px 3px 0;font-size:13px">'
        f'<span class="vital-dot" style="background:{lvl_color.get(lvl, "#999")}"></span>'
        f'{label}: <b>{val}</b></span>'
        for label, val, lvl in vitals)
    if attention:
        att_html = '<ul>' + ''.join(
            f'<li><span class="vital-dot" style="background:{lvl_color.get(lvl, "#999")}"></span>{label}</li>'
            for label, lvl in attention) + '</ul>'
    else:
        att_html = '<p class="good">All tracked vitals are in the green band. 所有追蹤指標皆在綠色區間。</p>'
    if stress14:
        tot_s, tot_r, last_label = stress14
        chip = f' · Latest day 最新一日：{last_label}' if last_label else ''
        stress_html = (f'<p style="font-size:14px">Stress total 壓力總時數 <b>{tot_s / 60:.1f} h</b> · '
                       f'Recovery total 恢復總時數 <b>{tot_r / 60:.1f} h</b><br/>'
                       f'Daily average 每日平均：stress {tot_s / n_days:.0f} min · recovery {tot_r / n_days:.0f} min{chip}</p>')
    else:
        stress_html = '<p class="muted">Not enough daily stress data in this window. 本區間每日壓力資料不足。</p>'
    if top_actions:
        act_html = '<ol>' + ''.join(
            f'<li><b>{dom}</b> <span class="muted">({risk})</span><br/>{act}</li>'
            for dom, risk, act in top_actions) + '</ol>'
    else:
        act_html = '<p class="muted">No recommendations available. 暫無建議。</p>'
    return f"""
    <div class="card">
      <div class="ring-wrap">{_ring_svg(sleep_mean, 'Sleep 睡眠')}{_ring_svg(readiness_mean, 'Readiness 準備度')}</div>
      <p style="margin-top:10px">{vitals_html}</p>
    </div>
    <h3>Needs Attention / 需要關注</h3>
    <div class="card">{att_html}</div>
    <h3>Stress &amp; Recovery (last 14 days) / 壓力與恢復（近 14 日）</h3>
    <div class="card">{stress_html}</div>
    <h3>Top Actions / 首要行動</h3>
    <div class="card">{act_html}</div>"""


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

    # Beta-test feedback: say plainly what was found, so a missing section
    # in the report is explainable without reading the code.
    _expected = ['dailyactivity.csv', 'dailysleep.csv', 'dailyreadiness.csv',
                 'dailyspo2.csv', 'dailystress.csv', 'daytimestress.csv',
                 'heartrate.csv', 'temperature.csv', 'sleepmodel.csv']
    _found = {Path(rel).name.lower() for rel in dataframes}
    _missing = [f for f in _expected if f not in _found]
    if _found:
        print(f"Found {len(_found)} CSV file(s) in {BASE}")
    else:
        print(f"WARNING: no CSV files found in {BASE} - the report will be empty. "
              "Point --input-dir at the folder with your Oura export.")
    if _missing:
        print(f"Missing (related sections will be skipped): {', '.join(_missing)}")

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
        (dailyactivity, 'day'), (dailysleep, 'day'), (readiness, 'day'), (dailyspo2, 'day'), (dailystress, 'day'), (sleepmodel, 'day')
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
        (sleepmodel, ['average_heart_rate', 'total_sleep_duration', 'deep_sleep_duration', 'rem_sleep_duration']),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = to_num(df[c])

    if 'spo2_percentage' in dailyspo2.columns:
        dailyspo2['spo2_avg'] = to_num(dailyspo2['spo2_percentage'].apply(lambda x: (parse_json_field(x) or {}).get('average')))

    # Per-view computation (extracted from the original main flow so the
    # Summary / Comparison / Trends tabs can each be re-rendered per period.
    # The full-data view below is computed exactly as before.)
    def compute_view(dailyactivity, dailysleep, readiness, dailyspo2, dailystress, sleepmodel):
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
        return {'sp_steps': sp_steps, 'sp_sleep_score': sp_sleep_score, 'sp_readiness': sp_readiness, 'sp_activity_score': sp_activity_score, 'sp_spo2': sp_spo2, 'sp_nightly_hr': sp_nightly_hr, 'sp_temp_dev': sp_temp_dev, 'sp_sleep_hours': sp_sleep_hours, 'sleep_under7': sleep_under7, 'spo2_below95': spo2_below95, 'nightly_hr_out': nightly_hr_out, 'temp_dev_over': temp_dev_over, 'bands_df': bands_df, 'compare_df': compare_df, 'sleep_level': sleep_level, 'hr_level': hr_level, 'spo2_level': spo2_level, 'temp_level': temp_level, 'sleep_score_level': sleep_score_level, 'readiness_level': readiness_level, 'steps_level': steps_level, 'lvl_text': lvl_text, 'nightly_hr': nightly_hr, 'sleep_mean': sleep_mean, 'hr_mean': hr_mean, 'spo2_mean': spo2_mean, 'temp_mean': temp_mean, 'sleep_score_mean': sleep_score_mean, 'readiness_mean': readiness_mean, 'steps_mean': steps_mean}
    def build_charts(dailyactivity, dailysleep, readiness, dailyspo2, dailystress, daystress, heartrate, temperature, sleepmodel, suffix):
        def save_chart_s(name, svg):
            return save_chart((suffix + name) if suffix else name, svg)
        # Trend charts
        charts: list[tuple[str, str]] = []

        def add_chart(df: pd.DataFrame, xcol: str, ycol: str, label: str, title: str, color: str, file: str):
            if xcol in df.columns and ycol in df.columns:
                d = df[[xcol, ycol]].dropna().sort_values(xcol)
                if not d.empty:
                    xs = [str(x)[:10] for x in d[xcol].tolist()]
                    ys = [float(y) for y in d[ycol].tolist()]
                    charts.append((label, save_chart_s(file, line_svg(xs, ys, title, color))))

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

        if 'ts' in daystress.columns and 'stress_value' in daystress.columns and 'recovery_value' in daystress.columns:
            ds = daystress.dropna(subset=['ts']).copy()
            ds['day_dt'] = ds['ts'].dt.tz_convert('UTC').dt.date
            ds = ds.groupby('day_dt', as_index=False)[['stress_value', 'recovery_value']].mean()
            if not ds.empty:
                xs = [str(x)[:10] for x in ds['day_dt'].tolist()]
                charts.append(('Stress vs Recovery / 壓力與恢復', save_chart_s('stress_recovery.svg', dual_line_svg(xs, ds['stress_value'].tolist(), ds['recovery_value'].tolist(), 'Daily Stress vs Recovery'))))

        # Daily temperature deviation (API has no skin-temperature time series;
        # daily deviation from readiness is the closest available signal).
        add_chart(readiness, 'day_dt', 'temperature_deviation', 'Temperature Deviation (Daily) / 體溫偏差（每日）', 'Daily Temperature Deviation', '#e07a5f', 'temp_dev.svg')

        # Daily stress vs recovery from dailystress.csv (API has no daytime
        # stress time series; daily aggregates are the closest available signal).
        if 'day_dt' in dailystress.columns and 'stress_high' in dailystress.columns and 'recovery_high' in dailystress.columns:
            ds = dailystress[['day_dt', 'stress_high', 'recovery_high']].dropna(subset=['day_dt']).sort_values('day_dt')
            if not ds.empty:
                xs = [str(x)[:10] for x in ds['day_dt'].tolist()]
                charts.append(('Stress vs Recovery (Daily) / 壓力與恢復（每日）', save_chart_s('stress_daily.svg', dual_line_svg(xs, (ds['stress_high'] / 60.0).tolist(), (ds['recovery_high'] / 60.0).tolist(), 'Daily Stress vs Recovery (minutes)'))))

        # Deep vs REM sleep minutes from sleepmodel.csv.
        if 'day_dt' in sleepmodel.columns and 'deep_sleep_duration' in sleepmodel.columns and 'rem_sleep_duration' in sleepmodel.columns:
            sm = sleepmodel[['day_dt', 'deep_sleep_duration', 'rem_sleep_duration']].dropna(subset=['day_dt']).sort_values('day_dt')
            if not sm.empty:
                xs = [str(x)[:10] for x in sm['day_dt'].tolist()]
                charts.append(('Deep vs REM Sleep (minutes) / 深睡與 REM（分鐘）', save_chart_s('sleep_stages.svg', dual_line_svg(xs, (sm['deep_sleep_duration'] / 60.0).tolist(), (sm['rem_sleep_duration'] / 60.0).tolist(), 'Deep vs REM Sleep (minutes)', 'Deep', 'REM'))))

        return charts

    _full = compute_view(dailyactivity, dailysleep, readiness, dailyspo2, dailystress, sleepmodel)
    sp_steps = _full['sp_steps']; sp_sleep_score = _full['sp_sleep_score']
    sp_readiness = _full['sp_readiness']; sp_activity_score = _full['sp_activity_score']
    sp_spo2 = _full['sp_spo2']; sp_nightly_hr = _full['sp_nightly_hr']
    sp_temp_dev = _full['sp_temp_dev']; sp_sleep_hours = _full['sp_sleep_hours']
    sleep_under7 = _full['sleep_under7']; spo2_below95 = _full['spo2_below95']
    nightly_hr_out = _full['nightly_hr_out']; temp_dev_over = _full['temp_dev_over']
    bands_df = _full['bands_df']; compare_df = _full['compare_df']
    sleep_level = _full['sleep_level']; hr_level = _full['hr_level']
    spo2_level = _full['spo2_level']; temp_level = _full['temp_level']
    sleep_score_level = _full['sleep_score_level']; readiness_level = _full['readiness_level']
    steps_level = _full['steps_level']
    lvl_text = _full['lvl_text']; nightly_hr = _full['nightly_hr']
    sleep_mean = _full['sleep_mean']; hr_mean = _full['hr_mean']; spo2_mean = _full['spo2_mean']
    temp_mean = _full['temp_mean']; sleep_score_mean = _full['sleep_score_mean']
    readiness_mean = _full['readiness_mean']; steps_mean = _full['steps_mean']
    charts = build_charts(dailyactivity, dailysleep, readiness, dailyspo2, dailystress, daystress, heartrate, temperature, sleepmodel, '')

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

    # Rule-based summary generator (no AI): chooses phrasing by risk pattern.
    def level_score(level: str) -> int:
        # green=0, yellow=1, red=2
        return {'green': 0, 'yellow': 1, 'red': 2}.get(level, 0)

    worst_level = max(
        level_score(sleep_level),
        level_score(hr_level),
        level_score(spo2_level),
        level_score(temp_level),
        level_score(readiness_level),
        level_score(sleep_score_level),
        level_score(steps_level),
    )

    sleep_priority = (sleep_level in ('red', 'yellow')) or (not math.isnan(sleep_under7) and sleep_under7 >= 30)
    recovery_priority = (readiness_level in ('red', 'yellow')) or (sleep_score_level in ('red', 'yellow'))
    cardio_priority = (hr_level == 'red') or (spo2_level == 'red')
    activity_priority = steps_level in ('red', 'yellow')

    opener_en = {
        0: "Your trends look relatively stable overall, and you already have a strong foundation.",
        1: "Your data shows a clear direction for improvement, and the good news is the next steps are practical.",
        2: "Your data suggests a few areas need earlier attention, and acting now can improve recovery and long-term health risk."
    }[worst_level]
    opener_zh = {
        0: "你的整體趨勢相對穩定，已經有不錯的健康基礎。",
        1: "你的數據已經指出明確的改善方向，而且下一步都很務實、可執行。",
        2: "你的數據顯示有幾個面向需要提早注意，現在開始調整有助於恢復與長期健康風險。"
    }[worst_level]

    if sleep_priority:
        p1_en = "Protect sleep consistency first: keep a fixed sleep window and aim for at least 7 hours on most nights."
        p1_zh = "先把睡眠規律穩住：固定入睡與起床時段，並讓多數夜晚睡眠達到至少 7 小時。"
    elif recovery_priority:
        p1_en = "Prioritize recovery rhythm: use low-readiness days for lighter load and keep hydration and sleep quality steady."
        p1_zh = "優先穩定恢復節奏：在準備度偏低時降低負荷，並維持補水與睡眠品質。"
    else:
        p1_en = "Keep your current routine and continue small, repeatable habits that support sleep and recovery."
        p1_zh = "請延續你目前的節奏，持續執行有助睡眠與恢復的小而穩定習慣。"

    if cardio_priority:
        p2_en = "For heart-rate and oxygen signals, focus on 3-7 day trends and pair the numbers with symptoms before making decisions."
        p2_zh = "心率與血氧訊號建議以 3-7 天趨勢判讀，並搭配症狀再做決策。"
    elif activity_priority:
        p2_en = "Support resilience with steady daytime movement and fewer prolonged sedentary blocks."
        p2_zh = "用穩定的日間活動量與減少久坐時段，來提升整體韌性與恢復能力。"
    else:
        p2_en = "Maintain balanced daytime activity and adjust training intensity on high-stress or low-readiness days."
        p2_zh = "維持平衡的白天活動量，並在高壓或低準備度日適度調整訓練強度。"

    summary_en_html = "<br/>".join([opener_en, p1_en, p2_en])
    summary_zh_html = "<br/>".join([opener_zh, p1_zh, p2_zh])

    # (trend charts are built by build_charts above, per view)
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
  <li>Lu MC. Keynote on Health Resilience. Taiwan, November 2026. (Framework for the resilience sections of this report.)</li>
</ul>
"""

    _top_actions = [(r.get('Domain / 面向', ''), r.get('Risk', ''), r.get('Action (2-4 weeks) / 建議行動（2-4 週）', '')) for r in rec_rows[:2]]
    _bg_path = BASE / 'resilience_background.html'
    background_html = ''
    if _bg_path.exists():
        background_html = ('<h2>Background: Health Resilience / 背景資料：健康韌性</h2>'
                           '<div class="card">' + _bg_path.read_text(encoding='utf-8') + '</div>')

    # --- Period views: Summary / Comparison / Trends each get a Period selector.
    # Weekly periods anchored at the first data day; default = latest period
    # (past week + current period days); "All data" keeps the full-window view.
    def _slice(df, d0, d1):
        if df is None or df.empty:
            return df
        if 'day_dt' in df.columns:
            return df[(df['day_dt'].dt.date >= d0) & (df['day_dt'].dt.date <= d1)]
        if 'ts' in df.columns:
            return df[(df['ts'].dt.date >= d0) & (df['ts'].dt.date <= d1)]
        return df

    def _period_selector(cls, idprefix, defs, extra_cls=None):
        opts = ''.join(f'<option value="{idprefix}-{k}">{label}</option>' for k, label, _ in defs)
        groups = f'.{cls}' + (f', .{extra_cls}' if extra_cls else '')
        onchange = (f"(document.querySelectorAll('{groups}').forEach(p=>p.style.display='none'),"
                    f"document.getElementById(this.value).style.display='block'"
                    + (f",document.getElementById(this.value+'-b').style.display='block'" if extra_cls else '') + ')')
        return f'<p><label>Period 期間：<select onchange="{onchange}">{opts}</select></label></p>'

    _axis = dailysleep if 'day_dt' in dailysleep.columns else dailystress
    _days = sorted(_axis['day_dt'].dt.date.unique().tolist()) if 'day_dt' in _axis.columns else []
    _weeks = [_days[i:i + 7] for i in range(0, len(_days), 7)]
    _view_defs = []
    if _days:
        _latest_rng = (_weeks[-2][0], _weeks[-1][-1]) if len(_weeks) >= 2 else (_days[0], _days[-1])
        _view_defs.append(('latest', 'Latest — past week + current period 最新（上週＋本期）', _latest_rng))
        _view_defs.append(('all', 'All data 全部期間', (_days[0], _days[-1])))
        for _i, _wk in enumerate(reversed(_weeks)):
            _view_defs.append((f'week{len(_weeks) - _i}', f'Week {len(_weeks) - _i} · {_wk[0]} to {_wk[-1]}', (_wk[0], _wk[-1])))

    _sum_blocks, _cmp_blocks, _bands_blocks, _trd_blocks = [], [], [], []
    for _vi, (_key, _label, (_d0, _d1)) in enumerate(_view_defs):
        _disp = 'block' if _vi == 0 else 'none'
        _sl = {n: _slice(f, _d0, _d1) for n, f in [('dailyactivity', dailyactivity), ('dailysleep', dailysleep),
               ('readiness', readiness), ('dailyspo2', dailyspo2), ('dailystress', dailystress), ('sleepmodel', sleepmodel)]}
        _v = _full if _key == 'all' else compute_view(_sl['dailyactivity'], _sl['dailysleep'], _sl['readiness'], _sl['dailyspo2'], _sl['dailystress'], _sl['sleepmodel'])
        _vit = [
            ('Sleep score 睡眠分數', fmt(_v['sp_sleep_score'].get('mean'), 0), _v['sleep_score_level']),
            ('Readiness 準備度', fmt(_v['sp_readiness'].get('mean'), 0), _v['readiness_level']),
            ('Sleep hours 睡眠時數', fmt(_v['sp_sleep_hours'].get('mean'), 1) + ' h', _v['sleep_level']),
            ('Night HR 夜間心率', fmt(_v['sp_nightly_hr'].get('mean'), 0) + ' bpm', _v['hr_level']),
            ('SpO2 血氧', fmt(_v['sp_spo2'].get('mean'), 1) + ' %', _v['spo2_level']),
            ('Temp dev 體溫偏差', fmt(_v['sp_temp_dev'].get('mean'), 2), _v['temp_level']),
            ('Steps 步數', fmt(_v['sp_steps'].get('mean'), 0), _v['steps_level']),
        ]
        _att = [(lbl, lvl) for (lbl, _vv, lvl) in _vit if lvl != 'green']
        _stv = None
        _nd = 14
        _dsv = _sl['dailystress']
        if not _dsv.empty and 'stress_high' in _dsv.columns and 'recovery_high' in _dsv.columns and 'day_dt' in _dsv.columns:
            _w = _dsv.dropna(subset=['day_dt']).sort_values('day_dt')
            if len(_w) >= 3:
                _stv = (float(_w['stress_high'].sum()), float(_w['recovery_high'].sum()),
                        str(_w['day_summary'].iloc[-1]) if 'day_summary' in _w.columns and pd.notna(_w['day_summary'].iloc[-1]) else '')
                _nd = max(1, len(_w))
        _sum_blocks.append(f'<div class="sum-period" id="sumperiod-{_key}" style="display:{_disp}">'
                           + summary_section(_v['sp_sleep_score'].get('mean'), _v['sp_readiness'].get('mean'), _vit, _att, _stv, _top_actions, window_text, _nd) + '</div>')
        _cmp_blocks.append(f'<div class="cmp-period" id="cmpperiod-{_key}" style="display:{_disp}">{table_html(_v["compare_df"])}</div>')
        _bands_blocks.append(f'<div class="cmp-period-b" id="cmpperiod-{_key}-b" style="display:{_disp}">{table_html(_v["bands_df"])}</div>')
        _vcharts = charts if _key == 'all' else build_charts(_sl['dailyactivity'], _sl['dailysleep'], _sl['readiness'], _sl['dailyspo2'], _sl['dailystress'], _slice(daystress, _d0, _d1), _slice(heartrate, _d0, _d1), _slice(temperature, _d0, _d1), _sl['sleepmodel'], f'_{_key}')
        _grid = ''.join([f'<div class="chart-item card"><h3>{t}</h3>{(ASSET_DIR / f).read_text(encoding="utf-8")}</div>' for t, f in _vcharts])
        _trd_blocks.append(f'<div class="trd-period" id="trdperiod-{_key}" style="display:{_disp}"><div class="chart-grid">{_grid}</div></div>')

    summary_html = _period_selector('sum-period', 'sumperiod', _view_defs) + ''.join(_sum_blocks)
    comparison_html = _period_selector('cmp-period', 'cmpperiod', _view_defs, 'cmp-period-b') + ''.join(_cmp_blocks)
    comparison_bands_html = ''.join(_bands_blocks)
    trends_html = _period_selector('trd-period', 'trdperiod', _view_defs) + ''.join(_trd_blocks)

    resilience_html = resilience_section(dailystress, dailysleep, sleepmodel)

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
    .tab-bar {{ display:flex; gap:6px; flex-wrap:wrap; margin:14px 0; }}
    .tab-btn {{ padding:7px 14px; border-radius:999px; border:1px solid var(--line); background:#fff; cursor:pointer; font-size:14px; }}
    .tab-btn.active {{ background:#333; color:#fff; border-color:#333; }}
    .ring-wrap {{ display:flex; gap:22px; flex-wrap:wrap; }}
    .vital-dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
    @media print {{ .tab-bar {{ display:none; }} .tab-panel {{ display:block !important; }} }}
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

    <div class="tab-bar"><button class="tab-btn active" onclick="(document.querySelectorAll('.tab-panel').forEach(p=>p.style.display='none'),document.getElementById('panel-summary').style.display='block',document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active')),this.classList.add('active'))">Summary 總覽</button><button class="tab-btn" onclick="(document.querySelectorAll('.tab-panel').forEach(p=>p.style.display='none'),document.getElementById('panel-resilience').style.display='block',document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active')),this.classList.add('active'))">Resilience 韌性</button><button class="tab-btn" onclick="(document.querySelectorAll('.tab-panel').forEach(p=>p.style.display='none'),document.getElementById('panel-comparison').style.display='block',document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active')),this.classList.add('active'))">Comparison 比較</button><button class="tab-btn" onclick="(document.querySelectorAll('.tab-panel').forEach(p=>p.style.display='none'),document.getElementById('panel-trends').style.display='block',document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active')),this.classList.add('active'))">Trends 趨勢</button><button class="tab-btn" onclick="(document.querySelectorAll('.tab-panel').forEach(p=>p.style.display='none'),document.getElementById('panel-notes').style.display='block',document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active')),this.classList.add('active'))">Notes 筆記</button><button class="tab-btn" onclick="(document.querySelectorAll('.tab-panel').forEach(p=>p.style.display='none'),document.getElementById('panel-actions').style.display='block',document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active')),this.classList.add('active'))">Actions 行動</button></div>
    <div class="tab-panel" id="panel-summary">{summary_html}</div>
    <div class="tab-panel" id="panel-resilience" style="display:none">{resilience_html}</div>
    <div class="tab-panel" id="panel-comparison" style="display:none">

    <h2>1) Analysis Approach / 分析方法</h2>
    <div class=\"card\">
      <ul>
        <li>Use general healthy standards first, then compare with your own data spread (median and P10-P90).</li>
        <li>Sleep, readiness, HR, SpO2 and temperature deviation are useful for early preventive signals.</li>
        <li>Single-day spikes are less important than multi-day trends (especially temperature and stress metrics).</li>
      </ul>
    </div>

    <h2>2) Preventive Comparison Table / 預防醫學比較表（含健康族群標準）</h2>
    <div class=\"card\">{comparison_html}</div>
    <p class=\"muted footnote-sm\">Color rule: <span class=\"danger\">Red</span> = high risk, <span class=\"warn\">Yellow</span> = mild caution, <span class=\"good\">Green</span> = relatively stable.</p>

    <h2>3) Oura Score Band Distribution / 分數區間分佈</h2>
    <div class=\"card\">{comparison_bands_html}
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

    </div>
    <div class="tab-panel" id="panel-notes" style="display:none">
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

    {background_html}
    </div>
    <div class="tab-panel" id="panel-trends" style="display:none">
    <h2>5) Trend Graphs / 趨勢圖</h2>
    {trends_html}

    </div>
    <div class="tab-panel" id="panel-actions" style="display:none">

    <h2>7) Medical Inference, Prediction, and Action Plan / 醫學推測、預測與行動建議</h2>
    <div class=\"card\">{rec_cards_html(rec_rows)}</div>
    <div class=\"card\" style=\"margin-top:10px;\">
      <p><b>Summary / 總結</b></p>
      <p>
        {summary_en_html}
      </p>
      <p>
        {summary_zh_html}
      </p>
    </div>
    <p class=\"muted footnote-sm\">Priority is ranked by: medical importance + feasibility + risk level (High/Moderate/Low). This section is decision-support, not a diagnosis.</p>

    <h2>8) References / 參考來源</h2>
    <div class=\"card\">{sources_html}</div>

    <p class=\"muted footnote-sm\">Medical disclaimer: This report was generated by an unofficial open-source tool for health education and early screening support only. It is not a medical device, and it does not diagnose, treat, cure, or prevent any disease or condition; nothing in it is medical advice, and no doctor-patient relationship is created. Wearable data can be inaccurate or incomplete, and the automated interpretations here may be wrong; they are provided &quot;as is&quot; without warranties of any kind, and the developer and contributors accept no liability for reliance on this report. Always seek the advice of a qualified clinician with any questions about a medical condition, and never disregard professional advice or delay seeking it because of this report. If you think you may have a medical emergency, call your local emergency number immediately. This tool is not affiliated with, endorsed by, or sponsored by Oura Health Oy. See DISCLAIMER.md in the project repository for the full disclaimer.</p>
    </div>
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
