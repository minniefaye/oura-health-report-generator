// Oura Health Report — the app version of the generated preventive health
// report. Opens on an Oura-app-style Summary tab (headline score rings,
// vitals at a glance, needs-attention callouts, top actions) with the full
// report sections one tap away in detail tabs: Resilience, Comparison,
// Trends, Notes, Actions — the Resilience tab sits right after Summary and
// shows only this app's Estimated Resilience calculation (level, trend,
// contributors, map). When a snapshot carries precomputed period views,
// the Summary, Comparison and Trends tabs each get a Period dropdown
// (latest / all data / weekly periods) like the Resilience tab's.
// Renders the newest stored snapshot by default and
// lets past snapshots be browsed from the history strip. All report
// content comes from the snapshot payload (server data); nothing
// person-specific is hard-coded here — the summary is derived from the
// same payload fields the detail tabs render.

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
CartesianGrid,
Line,
LineChart,
ReferenceArea,
ReferenceLine,
ResponsiveContainer,
Tooltip,
XAxis,
YAxis,
} from "recharts";
import { SafeAreaTopScrim } from "@hatch/space-sdk/client";
import { api, type ApiResponse } from "./api";

type SnapshotMeta = ApiResponse<typeof api, "listSnapshots">["snapshots"][number];
type SnapshotFull = NonNullable<ApiResponse<typeof api, "getSnapshot">["snapshot"]>;
type Payload = SnapshotFull["payload"];
type ComparisonRow = Payload["comparison"][number];
type RiskTone = Payload["comparison"][number]["risk"];
type PeriodView = NonNullable<Payload["periodViews"]>[number];

type TabId = "summary" | "comparison" | "trends" | "notes" | "actions" | "resilience";

const TABS: { id: TabId; en: string; zh: string }[] = [
  { id: "summary", en: "Summary", zh: "總覽" },
  { id: "resilience", en: "Resilience", zh: "韌性" },
  { id: "comparison", en: "Comparison", zh: "比較" },
  { id: "trends", en: "Trends", zh: "趨勢" },
  { id: "notes", en: "Notes", zh: "筆記" },
  { id: "actions", en: "Actions", zh: "行動" },
];

const RISK_TEXT: Record<RiskTone, string> = {
  red: "text-[var(--risk-red)]",
  yellow: "text-[var(--risk-amber)]",
  green: "text-[var(--risk-green)]",
};
const RISK_BADGE: Record<RiskTone, string> = {
  red: "border-[var(--risk-red)] text-[var(--risk-red)]",
  yellow: "border-[var(--risk-amber)] text-[var(--risk-amber)]",
  green: "border-[var(--risk-green)] text-[var(--risk-green)]",
};
const RISK_DOT: Record<RiskTone, string> = {
  red: "bg-[var(--risk-red)]",
  yellow: "bg-[var(--risk-amber)]",
  green: "bg-[var(--risk-green)]",
};

function SectionTitle({ children }: { children: string }) {
  return (
    <h2 className="mt-8 mb-3 text-xl font-semibold text-[var(--text)]">
      {children}
    </h2>
  );
}

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4 shadow-[0_6px_20px_rgba(30,43,67,0.06)] ${className}`}
    >
      {children}
    </div>
  );
}

function RiskBadge({ tone, label }: { tone: RiskTone; label: string }) {
  return (
    <span
      className={`inline-block rounded-full border px-2 py-0.5 text-xs font-bold whitespace-nowrap ${RISK_BADGE[tone]}`}
    >
      {label}
    </span>
  );
}

/* ---------------------------------- charts --------------------------------- */

const AXIS_TICK = { fontSize: 11, fill: "var(--dim)" };
const TOOLTIP_STYLE = {
  backgroundColor: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: 10,
  color: "var(--text)",
  fontSize: 12,
};

function shortDay(day: string): string {
  return day.slice(5); // MM-DD
}

/* ------------------------- daily-series period logic ------------------------ */
// The resilience estimate is computed over the maximum data available (at
// least 7 days required). Its daily detail is organized into weekly
// periods anchored at the first day of the dataset. The default view is
// the past week + the days of the current (latest, possibly partial)
// period; earlier weeks are reachable from the period dropdown.

type DailyRow = {
  date: string;
  index: number | null;
  stressRolling: number | null;
  recoveryRolling: number | null;
  stressMin: number;
  recoveryMin: number;
  label: string;
};

type DailySeriesRaw = [
  string,
  number | null,
  number | null,
  number | null,
  number,
  number,
  string,
][];

function parseDailySeries(raw: DailySeriesRaw): DailyRow[] {
  return raw.map(([date, index, sR, rR, sMin, rMin, label]) => ({
    date,
    index,
    stressRolling: sR,
    recoveryRolling: rR,
    stressMin: sMin,
    recoveryMin: rMin,
    label: label === "nan" ? "" : label,
  }));
}

type WeekPeriod = {
  weekNum: number;
  start: string;
  end: string;
  rows: DailyRow[];
};

function buildWeeks(rows: DailyRow[]): WeekPeriod[] {
  const weeks: WeekPeriod[] = [];
  for (let i = 0; i < rows.length; i += 7) {
    const slice = rows.slice(i, i + 7);
    const first = slice[0];
    const last = slice[slice.length - 1];
    if (first && last) {
      weeks.push({
        weekNum: weeks.length + 1,
        start: first.date,
        end: last.date,
        rows: slice,
      });
    }
  }
  return weeks;
}

// Default period: the past (last complete) week plus the days of the
// current period. With only one week of data, that week alone.
function defaultPeriodRows(rows: DailyRow[], weeks: WeekPeriod[]): DailyRow[] {
  if (weeks.length >= 2) {
    const prev = weeks[weeks.length - 2];
    const curr = weeks[weeks.length - 1];
    return [...(prev?.rows ?? []), ...(curr?.rows ?? [])];
  }
  return rows;
}

function levelForIndex(v: number): string {
  if (v < 40) return "Limited";
  if (v < 55) return "Adequate";
  if (v < 65) return "Solid";
  if (v < 80) return "Strong";
  return "Exceptional";
}

function SingleChart({
  title,
  data,
  color,
  formatY,
  zeroLine = false,
}: {
  title: string;
  data: Payload["charts"]["steps"];
  color: string;
  formatY?: (v: number) => string;
  zeroLine?: boolean;
}) {
  return (
    <Card>
      <h3 className="mb-2 text-sm font-semibold text-[var(--text)]">{title}</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="day"
              tickFormatter={shortDay}
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              minTickGap={24}
            />
            <YAxis
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={false}
              width={44}
              tickFormatter={formatY}
              domain={["auto", "auto"]}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              labelFormatter={(label) => String(label)}
              formatter={(value) => [String(value), title.split(" / ")[0] ?? ""]}
            />
            {zeroLine ? <ReferenceLine y={0} stroke="var(--dim)" strokeDasharray="4 4" /> : null}
            <Line
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={2.5}
              dot={{ r: 2.5, fill: color }}
              activeDot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

function DualChart({
  title,
  data,
  colorA,
  colorB,
  labelA,
  labelB,
}: {
  title: string;
  data: Payload["charts"]["stressRecovery"];
  colorA: string;
  colorB: string;
  labelA: string;
  labelB: string;
}) {
  return (
    <Card>
      <h3 className="mb-1 text-sm font-semibold text-[var(--text)]">{title}</h3>
      <div className="mb-1 flex gap-4 text-xs text-[var(--dim)]">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: colorA }} />
          {labelA}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: colorB }} />
          {labelB}
        </span>
      </div>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="day"
              tickFormatter={shortDay}
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              minTickGap={24}
            />
            <YAxis
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={false}
              width={44}
              domain={["auto", "auto"]}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              labelFormatter={(label) => String(label)}
              formatter={(value, name) => [String(value), name === "a" ? labelA : labelB]}
            />
            <Line type="monotone" dataKey="a" stroke={colorA} strokeWidth={2.5} dot={{ r: 2.5, fill: colorA }} activeDot={{ r: 4 }} />
            <Line type="monotone" dataKey="b" stroke={colorB} strokeWidth={2.5} dot={{ r: 2.5, fill: colorB }} activeDot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

/* ------------------------------ summary pieces ----------------------------- */

// Oura's own score bands, as stated in the payload's guideline text:
// 85-100 optimal, 70-84 good, 60-69 fair, below 60 pay attention. Colors
// match the band-distribution bars in the Comparison tab.
function scoreBand(value: number): { en: string; zh: string; color: string } {
  if (value >= 85) return { en: "Optimal", zh: "最佳", color: "#2a9d50" };
  if (value >= 70) return { en: "Good", zh: "良好", color: "#74c69d" };
  if (value >= 60) return { en: "Fair", zh: "尚可", color: "#e9c46a" };
  return { en: "Pay Attention", zh: "需注意", color: "#e63946" };
}

function ScoreRing({
  row,
}: {
  row: { metricEn: string; metricZh: string; mean: string };
}) {
  const value = Number.parseFloat(row.mean);
  const size = 136;
  const stroke = 10;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const known = Number.isFinite(value);
  const fraction = known ? Math.min(Math.max(value, 0), 100) / 100 : 0;
  const band = known ? scoreBand(value) : null;
  return (
    <div className="flex flex-col items-center text-center">
      <div
        className="relative"
        role="img"
        aria-label={`${row.metricEn}: ${row.mean} out of 100${band ? `, ${band.en}` : ""}`}
      >
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="var(--border)"
            strokeWidth={stroke}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={band?.color ?? "var(--dim)"}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * (1 - fraction)}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-bold [font-variant-numeric:tabular-nums] text-[var(--text)]">
            {row.mean}
          </span>
          <span className="text-xs text-[var(--dim)]">/ 100</span>
        </div>
      </div>
      <p className="mt-2 font-semibold text-[var(--text)]">
        {row.metricEn} <span className="text-[var(--dim)]">/ {row.metricZh}</span>
      </p>
      {band ? (
        <p className="text-sm font-semibold" style={{ color: band.color }}>
          {band.en} · {band.zh}
        </p>
      ) : null}
      <p className="text-xs text-[var(--dim)]">window average / 本期平均</p>
    </div>
  );
}

// The vitals strip picks these comparison rows, in this order, when present.
const VITAL_METRICS = [
  "Main sleep duration",
  "Night average heart rate",
  "Blood oxygen (SpO2)",
  "Temperature deviation",
  "Daily steps",
];

/* ------------------------------ period picker ------------------------------ */
// Shared Period dropdown for the Summary, Comparison and Trends tabs,
// driven by the payload's precomputed period views (latest / all data /
// weekly periods). Rendered only when the snapshot carries period views;
// each tab keeps its own selection.

const RISK_LABEL_BY_TONE: Record<RiskTone, string> = {
  red: "High Risk / 高風險",
  yellow: "Mild Caution / 輕度注意",
  green: "Good / 穩定",
};

function PeriodPicker({
  id,
  views,
  value,
  onChange,
}: {
  id: string;
  views: PeriodView[];
  value: string;
  onChange: (key: string) => void;
}) {
  const selected = views.find((v) => v.key === value) ?? views[0];
  return (
    <div className="mt-5">
      <label htmlFor={id} className="text-sm font-semibold text-[var(--text)]">
        Period / 期間
      </label>
      <select
        id={id}
        aria-label="Period"
        value={selected?.key ?? ""}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1.5 block min-h-11 w-full rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)]"
      >
        {views.map((v) => (
          <option key={v.key} value={v.key}>
            {v.labelEn} · {v.labelZh}
          </option>
        ))}
      </select>
      {selected ? (
        <p className="mt-1.5 text-xs text-[var(--dim)]">
          Showing / 顯示期間：{selected.start} → {selected.end}
        </p>
      ) : null}
    </div>
  );
}

function SummaryView({
  snapshot,
  onGoTab,
}: {
  snapshot: SnapshotFull;
  onGoTab: (tab: TabId) => void;
}) {
  const p = snapshot.payload;
  // Period views: when the snapshot carries precomputed period views, the
  // rings, vitals, needs-attention list and stress card describe the
  // selected period (default: the first view, "latest"); the top priority
  // actions stay global. Without views, everything derives from the
  // top-level payload fields exactly as before.
  const periodViews = p.periodViews ?? [];
  const [periodKey, setPeriodKey] = useState<string | null>(null);
  const view = periodViews.find((v) => v.key === periodKey) ?? periodViews[0];
  const scoreRows: { metricEn: string; metricZh: string; mean: string }[] =
    view
      ? [
          ...(view.summary.sleepScoreMean != null
            ? [
                {
                  metricEn: "Sleep score",
                  metricZh: "睡眠分數",
                  mean: String(view.summary.sleepScoreMean),
                },
              ]
            : []),
          ...(view.summary.readinessMean != null
            ? [
                {
                  metricEn: "Readiness score",
                  metricZh: "準備度分數",
                  mean: String(view.summary.readinessMean),
                },
              ]
            : []),
        ]
      : p.comparison.filter((r) => r.unit === "0-100");
  type VitalItem = {
    key: string;
    en: string;
    zh: string;
    value: string;
    sub: string;
    tone: RiskTone;
  };
  const vitals: VitalItem[] = view
    ? view.summary.vitals.map((v) => ({
        key: v.labelEn,
        en: v.labelEn,
        zh: v.labelZh,
        value: v.value,
        sub: "period avg / 本期間平均",
        tone: v.level,
      }))
    : VITAL_METRICS.map((name) =>
        p.comparison.find((r) => r.metricEn === name),
      )
        .filter((r): r is ComparisonRow => r != null)
        .map((row) => ({
          key: row.metricEn,
          en: row.metricEn,
          zh: row.metricZh,
          value: row.mean,
          sub: `${row.unit} · avg / 平均`,
          tone: row.risk,
        }));
  type AttentionItem = {
    key: string;
    en: string;
    zh: string;
    tone: RiskTone;
    badge: string;
    interpretation?: string;
  };
  const attention: AttentionItem[] = view
    ? view.summary.attention.map((a) => ({
        key: a.labelEn,
        en: a.labelEn,
        zh: a.labelZh,
        tone: a.level,
        badge: RISK_LABEL_BY_TONE[a.level],
      }))
    : p.comparison
        .filter((r) => r.risk !== "green")
        .map((row) => ({
          key: row.metricEn,
          en: row.metricEn,
          zh: row.metricZh,
          tone: row.risk,
          badge: row.riskLabel,
          interpretation: row.interpretation,
        }));
  const topActions = p.recommendations.slice(0, 2);
  // Stress & recovery at a glance. With period views, it totals the
  // selected period from the view's precomputed stress block. Otherwise it
  // describes the DEFAULT period — the past week + the days of the current
  // period — computed from the full-dataset daily series when the snapshot
  // carries one; older snapshots fall back to the report window's
  // stress/recovery chart series.
  const viewStress = view?.summary.stress ?? null;
  const dailyRows = p.resilience?.dailySeries
    ? parseDailySeries(p.resilience.dailySeries as DailySeriesRaw)
    : null;
  const stressPeriodRows =
    dailyRows && dailyRows.length > 0
      ? defaultPeriodRows(dailyRows, buildWeeks(dailyRows))
      : null;
  const stressData = p.charts.stressRecovery;
  const stressCount = view
    ? (viewStress?.days ?? 0)
    : stressPeriodRows
      ? stressPeriodRows.length
      : stressData.length;
  const hasStress = view
    ? viewStress != null && (viewStress.stressMin > 0 || viewStress.recoveryMin > 0)
    : stressPeriodRows
      ? stressPeriodRows.some((d) => d.stressMin > 0 || d.recoveryMin > 0)
      : stressData.length > 0 && stressData.some((d) => d.a > 0 || d.b > 0);
  const stressTotal = view
    ? (viewStress?.stressMin ?? 0)
    : stressPeriodRows
      ? stressPeriodRows.reduce((s, d) => s + d.stressMin, 0)
      : stressData.reduce((s, d) => s + d.a, 0);
  const recoveryTotal = view
    ? (viewStress?.recoveryMin ?? 0)
    : stressPeriodRows
      ? stressPeriodRows.reduce((s, d) => s + d.recoveryMin, 0)
      : stressData.reduce((s, d) => s + d.b, 0);
  const stressAvg = view
    ? (viewStress?.avgStressMin ?? 0)
    : Math.round(stressTotal / Math.max(stressCount, 1));
  const recoveryAvg = view
    ? (viewStress?.avgRecoveryMin ?? 0)
    : Math.round(recoveryTotal / Math.max(stressCount, 1));
  const stressPeriodLabel = stressPeriodRows
    ? `${stressPeriodRows[0]?.date ?? ""} → ${stressPeriodRows[stressPeriodRows.length - 1]?.date ?? ""}`
    : null;
  const stressHeader = view
    ? `Period totals · ${stressCount} days · ${view.start} → ${view.end} / 本期間總計`
    : stressPeriodLabel
      ? `Past week + current period · ${stressCount} days · ${stressPeriodLabel} / 上週＋本期`
      : `Window totals · ${stressCount} days / 本期總計`;
  const stressMax = Math.max(stressTotal, recoveryTotal, 1);
  const stressBalance =
    recoveryTotal > 0
      ? `${(stressTotal / recoveryTotal).toFixed(1)} : 1`
      : stressTotal > 0
        ? "stress only"
        : "—";
  const latestStressSummary = view
    ? viewStress?.latestLabel || undefined
    : dailyRows
      ? [...dailyRows].reverse().find((s) => s.label.trim() !== "")?.label
      : [...(p.stressDays ?? [])]
          .reverse()
          .find((s) => s.summary.trim() !== "")?.summary;

  return (
    <div>
      <p className="mt-5 text-sm text-[var(--dim)]">
        {view
          ? `This period at a glance / 本期間總覽 · ${view.start} → ${view.end}`
          : `This window at a glance / 本期總覽 · ${snapshot.windowStart} → ${snapshot.windowEnd}`}
      </p>

      {periodViews.length > 0 ? (
        <PeriodPicker
          id="summary-period"
          views={periodViews}
          value={view?.key ?? ""}
          onChange={setPeriodKey}
        />
      ) : null}

      {/* Headline scores, Oura style */}
      {scoreRows.length > 0 ? (
        <Card className="mt-3">
          <div className="flex flex-wrap justify-center gap-x-10 gap-y-6 py-2">
            {scoreRows.map((row) => (
              <ScoreRing key={row.metricEn} row={row} />
            ))}
          </div>
        </Card>
      ) : null}

      {/* Vitals strip */}
      {vitals.length > 0 ? (
        <>
          <SectionTitle>Vitals / 關鍵數據</SectionTitle>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            {vitals.map((item) => (
              <Card key={item.key} className="p-3.5">
                <p className="flex items-start justify-between gap-1 text-xs text-[var(--dim)]">
                  <span>
                    {item.en}
                    <br />
                    {item.zh}
                  </span>
                  <span className={`mt-1 inline-block h-2.5 w-2.5 shrink-0 rounded-full ${RISK_DOT[item.tone]}`} />
                </p>
                <p className="mt-1.5 text-2xl font-bold [font-variant-numeric:tabular-nums] text-[var(--text)]">
                  {item.value}
                </p>
                <p className="text-xs text-[var(--dim)]">{item.sub}</p>
              </Card>
            ))}
          </div>
        </>
      ) : null}

      {/* Stress & recovery at a glance — tap through to the Resilience tab */}
      {hasStress ? (
        <>
          <SectionTitle>Stress &amp; Recovery / 壓力與恢復</SectionTitle>
          <button
            type="button"
            onClick={() => onGoTab("resilience")}
            className="block w-full rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4 text-left shadow-[0_6px_20px_rgba(30,43,67,0.06)]"
          >
            <span className="flex items-center justify-between gap-2">
              <span className="text-xs text-[var(--dim)]">{stressHeader}</span>
              {latestStressSummary ? (
                <span className="inline-flex items-center gap-1.5">
                  <span className="text-[11px] text-[var(--dim)]">
                    Latest day / 最近一日
                  </span>
                  <StressSummaryChip summary={latestStressSummary} />
                </span>
              ) : null}
            </span>
            <span className="mt-3 block space-y-2.5">
              <span className="flex items-center gap-2">
                <span className="w-20 shrink-0 text-xs text-[var(--dim)]">
                  Stress 壓力
                </span>
                <span className="h-2.5 flex-1 overflow-hidden rounded-full bg-[var(--border)]">
                  <span
                    className="block h-full rounded-full bg-[var(--accent)]"
                    style={{ width: `${(stressTotal / stressMax) * 100}%` }}
                  />
                </span>
                <span className="w-28 shrink-0 text-right text-xs [font-variant-numeric:tabular-nums] text-[var(--text)]">
                  {stressTotal.toLocaleString()} min · {stressAvg}/day
                </span>
              </span>
              <span className="flex items-center gap-2">
                <span className="w-20 shrink-0 text-xs text-[var(--dim)]">
                  Recovery 恢復
                </span>
                <span className="h-2.5 flex-1 overflow-hidden rounded-full bg-[var(--border)]">
                  <span
                    className="block h-full rounded-full bg-[var(--risk-green)]"
                    style={{ width: `${(recoveryTotal / stressMax) * 100}%` }}
                  />
                </span>
                <span className="w-28 shrink-0 text-right text-xs [font-variant-numeric:tabular-nums] text-[var(--text)]">
                  {recoveryTotal.toLocaleString()} min · {recoveryAvg}/day
                </span>
              </span>
            </span>
            <span className="mt-3 block text-xs text-[var(--dim)]">
              Balance 平衡 = stress : recovery ·{" "}
              <span className="font-semibold text-[var(--text)]">
                {stressBalance}
              </span>{" "}
              — tap for the day-by-day picture / 點按查看每日明細 →
            </span>
          </button>
        </>
      ) : null}

      {/* Needs attention — tap through to the Comparison tab */}
      {attention.length > 0 ? (
        <>
          <SectionTitle>Needs Attention / 需要注意</SectionTitle>
          <div className="grid gap-3">
            {attention.map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => onGoTab("comparison")}
                className="flex w-full items-center gap-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] p-4 text-left shadow-[0_6px_20px_rgba(30,43,67,0.06)]"
              >
                <span className={`inline-block h-3 w-3 shrink-0 rounded-full ${RISK_DOT[item.tone]}`} />
                <span className="min-w-0 flex-1">
                  <span className="block font-semibold text-[var(--text)]">
                    {item.en} <span className="text-[var(--dim)]">/ {item.zh}</span>
                  </span>
                  {item.interpretation ? (
                    <span className="block truncate text-sm text-[var(--dim)]">
                      {item.interpretation}
                    </span>
                  ) : null}
                </span>
                <RiskBadge tone={item.tone} label={item.badge} />
              </button>
            ))}
          </div>
        </>
      ) : null}

      {/* Top priority actions */}
      {topActions.length > 0 ? (
        <>
          <SectionTitle>Top Priorities / 首要行動</SectionTitle>
          <div className="grid gap-3">
            {topActions.map((rec) => (
              <Card key={rec.priority}>
                <div className="flex items-center justify-between gap-2">
                  <span className="rounded-full border border-[#9ad6b3] bg-[#e6f6ec] px-2 py-0.5 text-xs font-bold text-[#15693f] dark:border-[#2c5c40] dark:bg-[#173325] dark:text-[#7fe0a8]">
                    Priority {rec.priority}
                  </span>
                  <RiskBadge tone={rec.risk} label={rec.riskLabel} />
                </div>
                <h3 className="mt-2 text-base font-semibold text-[var(--text)]">
                  {rec.domainEn} <span className="text-[var(--dim)]">/ {rec.domainZh}</span>
                </h3>
                <p className="mt-1 text-sm">
                  <b>Action 2-4 Weeks / 行動建議：</b>{" "}
                  <span className={rec.risk === "green" ? "" : `font-semibold ${RISK_TEXT[rec.risk]}`}>
                    {rec.action}
                  </span>
                </p>
              </Card>
            ))}
          </div>
          <button
            type="button"
            onClick={() => onGoTab("actions")}
            className="mt-3 min-h-11 w-full rounded-full border border-[var(--accent)] px-4 py-2 text-sm font-semibold text-[var(--accent)]"
          >
            See all actions / 查看全部行動建議 →
          </button>
        </>
      ) : null}
    </div>
  );
}

/* ------------------------------ detail sections ---------------------------- */

function ComparisonSection({
  comparison,
}: {
  comparison: Payload["comparison"];
}) {
  return (
    <>
      <SectionTitle>Preventive Comparison Table / 預防醫學比較表（含健康族群標準）</SectionTitle>

      {/* Phone: one card per metric */}
      <div className="grid gap-3 md:hidden">
        {comparison.map((row) => (
          <Card key={row.metricEn}>
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className="font-semibold text-[var(--text)]">
                  {row.metricEn} <span className="text-[var(--dim)]">/ {row.metricZh}</span>
                </p>
                <p className="text-xs text-[var(--dim)]">{row.unit}</p>
              </div>
              <RiskBadge tone={row.risk} label={row.riskLabel} />
            </div>
            <p className="mt-2 text-sm">
              <span className="text-[var(--dim)]">Your mean: </span>
              <span className={`text-lg font-bold [font-variant-numeric:tabular-nums] ${RISK_TEXT[row.risk]}`}>
                {row.mean}
              </span>{" "}
              <span className="text-[var(--dim)]">{row.unit}</span>
            </p>
            <dl className="mt-2 space-y-1.5 text-sm">
              <div>
                <dt className="text-xs text-[var(--dim)]">Your data spread</dt>
                <dd>{row.spread}</dd>
              </div>
              <div>
                <dt className="text-xs text-[var(--dim)]">Healthy population standard</dt>
                <dd>{row.standard}</dd>
              </div>
              <div>
                <dt className="text-xs text-[var(--dim)]">Guideline / target</dt>
                <dd>{row.guideline}</dd>
              </div>
              <div>
                <dt className="text-xs text-[var(--dim)]">Out-of-range rate</dt>
                <dd className={`font-semibold ${RISK_TEXT[row.risk]}`}>{row.outOfRange}</dd>
              </div>
              <div>
                <dt className="text-xs text-[var(--dim)]">Easy interpretation</dt>
                <dd>{row.interpretation}</dd>
              </div>
            </dl>
          </Card>
        ))}
      </div>

      {/* Desktop: the report's table */}
      <Card className="hidden overflow-x-auto p-0 md:block">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="bg-[var(--bg)] text-left">
              {["Metric / 指標", "Risk level", "Unit", "Your mean", "Your data spread", "Healthy population standard", "Guideline / target", "Out-of-range rate", "Easy interpretation"].map(
                (h) => (
                  <th key={h} className="border-b border-[var(--border)] px-2.5 py-2 font-semibold">
                    {h}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {comparison.map((row) => (
              <tr key={row.metricEn} className="align-top">
                <td className="border-b border-[var(--border)] px-2.5 py-2 font-medium whitespace-nowrap">
                  {row.metricEn}
                  <br />
                  <span className="text-[var(--dim)]">{row.metricZh}</span>
                </td>
                <td className="border-b border-[var(--border)] px-2.5 py-2">
                  <RiskBadge tone={row.risk} label={row.riskLabel} />
                </td>
                <td className="border-b border-[var(--border)] px-2.5 py-2 whitespace-nowrap">{row.unit}</td>
                <td className={`border-b border-[var(--border)] px-2.5 py-2 font-bold [font-variant-numeric:tabular-nums] ${RISK_TEXT[row.risk]}`}>
                  {row.mean}
                </td>
                <td className="border-b border-[var(--border)] px-2.5 py-2">{row.spread}</td>
                <td className="border-b border-[var(--border)] px-2.5 py-2">{row.standard}</td>
                <td className="border-b border-[var(--border)] px-2.5 py-2">{row.guideline}</td>
                <td className={`border-b border-[var(--border)] px-2.5 py-2 font-semibold ${RISK_TEXT[row.risk]}`}>
                  {row.outOfRange}
                </td>
                <td className="border-b border-[var(--border)] px-2.5 py-2">{row.interpretation}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <p className="mt-2 text-sm text-[var(--dim)]">
        Color rule: <span className="font-bold text-[var(--risk-red)]">Red</span> = high risk,{" "}
        <span className="font-bold text-[var(--risk-amber)]">Yellow</span> = mild caution,{" "}
        <span className="font-bold text-[var(--risk-green)]">Green</span> = relatively stable.
      </p>
    </>
  );
}

const BAND_SEGMENTS = [
  { key: "optimal", label: "Optimal (85-100)", color: "#2a9d50" },
  { key: "good", label: "Good (70-84)", color: "#74c69d" },
  { key: "fair", label: "Fair (60-69)", color: "#e9c46a" },
  { key: "attention", label: "Pay attention (0-59)", color: "#e63946" },
] as const;

function BandsSection({
  bands,
  bandEvalZh,
}: {
  bands: Payload["bands"];
  bandEvalZh: string;
}) {
  return (
    <>
      <SectionTitle>Oura Score Band Distribution / 分數區間分佈</SectionTitle>
      <div className="grid gap-3">
        {bands.map((band) => (
          <Card key={band.nameEn}>
            <div className="flex items-baseline justify-between gap-2">
              <p className="font-semibold text-[var(--text)]">
                {band.nameEn} <span className="text-[var(--dim)]">/ {band.nameZh}</span>
              </p>
              <p className="text-xs text-[var(--dim)]">{band.total} days total</p>
            </div>
            <div
              className="mt-3 flex h-6 w-full overflow-hidden rounded-full border border-[var(--border)]"
              role="img"
              aria-label={`${band.nameEn} band distribution: ${BAND_SEGMENTS.map(
                (s) => `${s.label} ${band[s.key].count} days (${band[s.key].pct}%)`,
              ).join(", ")}`}
            >
              {BAND_SEGMENTS.map((s) =>
                band[s.key].pct > 0 ? (
                  <div
                    key={s.key}
                    style={{ width: `${band[s.key].pct}%`, background: s.color }}
                    className="h-full"
                  />
                ) : null,
              )}
            </div>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--dim)]">
              {BAND_SEGMENTS.map((s) => (
                <span key={s.key} className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
                  {s.label}:{" "}
                  <span className="font-semibold text-[var(--text)] [font-variant-numeric:tabular-nums]">
                    {band[s.key].count} ({band[s.key].pct}%)
                  </span>
                </span>
              ))}
            </div>
            <p className="mt-2 text-sm text-[var(--dim)]">
              {band.noteEn}
              <br />
              {band.noteZh}
            </p>
          </Card>
        ))}
      </div>
      <p className="mt-3 text-sm text-[var(--dim)]">
        <span className="font-semibold text-[var(--text)]">How this is evaluated / 怎麼評估：</span>{" "}
        {bandEvalZh}
      </p>
    </>
  );
}

function NotesSection({ payload }: { payload: Payload }) {
  return (
    <>
      <SectionTitle>Data-Driven Clinical Notes / 醫學解讀筆記</SectionTitle>
      <Card>
        <ul className="list-disc space-y-1.5 pl-5 text-sm">
          {payload.notesEn.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
        <div className="mt-4 space-y-2 border-t border-[var(--border)] pt-3 text-sm">
          {payload.notesZh.map((n) => (
            <p key={n}>{n}</p>
          ))}
        </div>
      </Card>
    </>
  );
}

function ChartsSection({ charts }: { charts: Payload["charts"] }) {
  const c = charts;
  return (
    <>
      <SectionTitle>Trend Graphs / 趨勢圖</SectionTitle>
      <div className="grid gap-3 lg:grid-cols-2">
        <SingleChart title="Daily Steps / 每日步數" data={c.steps} color="#ff6b8b" formatY={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : String(v))} />
        <SingleChart title="Sleep Score / 睡眠分數" data={c.sleepScore} color="#5e60ce" />
        <SingleChart title="Readiness Score / 準備度分數" data={c.readiness} color="#2a9d8f" />
        <SingleChart title="Heart Rate (Daily Avg) / 心率（日均）" data={c.heartRate} color="#e63946" />
        <SingleChart title="Temperature Deviation / 體溫偏差（每日）" data={c.tempDeviation} color="#e07a5f" zeroLine />
        <DualChart title="Stress vs Recovery (Daily) / 壓力與恢復（每日）" data={c.stressRecovery} colorA="#e07a5f" colorB="#2a9d8f" labelA="Stress (min)" labelB="Recovery (min)" />
        <DualChart title="Deep vs REM Sleep (minutes) / 深睡與 REM（分鐘）" data={c.deepRem} colorA="#5e60ce" colorB="#ff6b8b" labelA="Deep" labelB="REM" />
      </div>
    </>
  );
}

function RecommendationsSection({ payload }: { payload: Payload }) {
  return (
    <>
      <SectionTitle>Medical Inference, Prediction, and Action Plan / 醫學推測、預測與行動建議</SectionTitle>
      <div className="grid gap-3">
        {payload.recommendations.map((rec) => (
          <Card key={rec.priority}>
            <div className="flex items-center justify-between gap-2">
              <span className="rounded-full border border-[#9ad6b3] bg-[#e6f6ec] px-2 py-0.5 text-xs font-bold text-[#15693f] dark:border-[#2c5c40] dark:bg-[#173325] dark:text-[#7fe0a8]">
                Priority {rec.priority}
              </span>
              <RiskBadge tone={rec.risk} label={rec.riskLabel} />
            </div>
            <h3 className="mt-2 text-base font-semibold text-[var(--text)]">
              {rec.domainEn} <span className="text-[var(--dim)]">/ {rec.domainZh}</span>
            </h3>
            <div className="mt-1 space-y-1.5 text-sm">
              <p>
                <b>Data Suggests / 資料推測：</b> {rec.dataSuggests}
              </p>
              <p>
                <b>Medical Meaning / 醫學意義：</b> {rec.medicalMeaning}
              </p>
              <p>
                <b>Action 2-4 Weeks / 行動建議：</b>{" "}
                <span className={rec.risk === "green" ? "" : `font-semibold ${RISK_TEXT[rec.risk]}`}>
                  {rec.action}
                </span>
              </p>
              <p>
                <b>Track / 追蹤：</b> {rec.track}
              </p>
            </div>
          </Card>
        ))}
      </div>

      <Card className="mt-4">
        <h3 className="text-base font-semibold text-[var(--text)]">Summary / 總結</h3>
        <ul className="mt-2 list-disc space-y-1.5 pl-5 text-sm">
          {payload.summaryEn.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
        <ul className="mt-2 list-disc space-y-1.5 pl-5 text-sm">
          {payload.summaryZh.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
        <p className="mt-3 text-xs text-[var(--dim)]">{payload.priorityNote}</p>
      </Card>
    </>
  );
}

function ReferencesSection({ payload }: { payload: Payload }) {
  return (
    <>
      <SectionTitle>References / 參考來源</SectionTitle>
      <Card>
        <ul className="list-disc space-y-1 pl-5 text-sm">
          {payload.references.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
        <p className="mt-4 border-t border-[var(--border)] pt-3 text-xs leading-relaxed text-[var(--dim)]">
          <b className="text-[var(--text)]">Medical disclaimer:</b> {payload.disclaimer}
        </p>
      </Card>
    </>
  );
}

// Oura day_summary labels: English stays primary, with a Chinese gloss and a
// tone color per label. Unknown labels render English-only in a neutral tone.
const STRESS_SUMMARY_META: Record<string, { zh: string; color: string }> = {
  stressful: { zh: "高壓", color: "var(--risk-red)" },
  engaged: { zh: "投入", color: "var(--risk-amber)" },
  relaxed: { zh: "放鬆", color: "var(--risk-green)" },
  restored: { zh: "已恢復", color: "var(--risk-green)" },
  normal: { zh: "正常", color: "var(--dim)" },
};

function StressSummaryChip({ summary }: { summary: string }) {
  const meta = STRESS_SUMMARY_META[summary.toLowerCase()];
  const color = meta?.color ?? "var(--dim)";
  return (
    <span
      className="inline-block rounded-full px-1.5 py-0.5 text-[10px] leading-tight font-medium whitespace-nowrap"
      style={{
        color,
        backgroundColor: `color-mix(in srgb, ${color} 14%, transparent)`,
      }}
    >
      {summary}
      {meta ? ` ${meta.zh}` : ""}
    </span>
  );
}

/* --------------------- estimated resilience (computed) --------------------- */
// A transparent estimate computed by Muse from the snapshot's own ring
// data — NOT Oura's official score (its algorithm is proprietary and its
// API endpoint is not authorized on the user's connection). Labeled as an
// estimate everywhere it appears. Styled after Oura's own Resilience
// screens (Trend / Map / contributors), which the user shared screenshots
// of. Renders only when the snapshot payload carries a `resilience` field.

type ResilienceEstimate = NonNullable<Payload["resilience"]>;

// Oura's five resilience levels, bottom to top, with the index bands from
// the estimate's method and a tone color per level.
const RESILIENCE_BANDS: {
  level: string;
  y1: number;
  y2: number;
  fill: string;
  badge: string;
}[] = [
  { level: "Limited", y1: 0, y2: 40, fill: "#e63946", badge: "#e63946" },
  { level: "Adequate", y1: 40, y2: 55, fill: "#e9c46a", badge: "#b26a00" },
  { level: "Solid", y1: 55, y2: 65, fill: "#74c69d", badge: "#2a9d50" },
  { level: "Strong", y1: 65, y2: 80, fill: "#2a9d50", badge: "#2a9d50" },
  { level: "Exceptional", y1: 80, y2: 100, fill: "#1b4332", badge: "#1b4332" },
];

function resilienceLevelColor(level: string): string {
  return (
    RESILIENCE_BANDS.find((b) => b.level === level)?.badge ?? "var(--dim)"
  );
}

// Contributor rating tones, mirroring the app's risk colors: "Pay
// attention" and "High" (stress load) are the warm alert tone, "Fair" is
// neutral, and the favorable ratings read green.
function contributorRatingColor(rating: string): string {
  if (rating === "Pay attention" || rating === "High") return "#d96c4a";
  if (rating === "Fair" || rating === "Moderate") return "var(--dim)";
  return "var(--risk-green)";
}

function ResilienceTrendChart({
  data,
}: {
  data: ResilienceEstimate["trend"];
}) {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 6, right: 76, bottom: 0, left: 0 }}
        >
          {RESILIENCE_BANDS.map((band) => (
            <ReferenceArea
              key={band.level}
              y1={band.y1}
              y2={band.y2}
              fill={band.fill}
              fillOpacity={0.13}
              stroke="none"
              label={{
                value: band.level,
                position: "insideRight",
                fontSize: 11,
                fill: "var(--dim)",
              }}
            />
          ))}
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="day"
            tickFormatter={shortDay}
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            minTickGap={24}
          />
          <YAxis
            domain={[0, 100]}
            ticks={[0, 20, 40, 55, 65, 80, 100]}
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={false}
            width={32}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={(label) => String(label)}
            formatter={(value, _name, item) => {
              const level = (item?.payload as { level?: string } | undefined)
                ?.level;
              return [
                `${String(value)}${level ? ` · ${level}` : ""}`,
                "Estimated index",
              ];
            }}
          />
          <Line
            type="monotone"
            dataKey="index"
            stroke="var(--text)"
            strokeWidth={2.5}
            dot={{ r: 3, fill: "var(--text)" }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// Oura Map view: one point per day — that day's 14-day rolling average,
// x = stress minutes, y = recovery minutes — connected in date order with
// the latest day highlighted (pink dot). Like Oura's own map, four dashed
// diagonal lines divide the plot into the five Resilience level bands
// (Exceptional / Strong / Solid / Adequate / Limited, reading down), and a
// solid center cross marks the axes: vertical at mid-stress, horizontal
// at the Low stress / High stress label height (~38% of the plot).
// Fixed absolute scales (stress 0–360 min, recovery 0–240 min) keep
// positions comparable across windows and match how Oura's map reads;
// points outside a range clip to the edge.
function formatMapDayLabel(day: string): string {
  const d = new Date(`${day}T00:00:00`);
  if (Number.isNaN(d.getTime())) return day;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function ResilienceMap({
  points,
}: {
  points: ResilienceEstimate["map"];
}) {
  if (points.length === 0) return null;
  const width = 340;
  const height = 264;
  const pad = { top: 26, right: 14, bottom: 26, left: 14 };
  const MAX_STRESS = 360;
  const MAX_RECOVERY = 240;
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const clampStress = (v: number) => Math.min(Math.max(v, 0), MAX_STRESS);
  const clampRecovery = (v: number) => Math.min(Math.max(v, 0), MAX_RECOVERY);
  const x = (v: number) =>
    pad.left + (clampStress(v) / MAX_STRESS) * plotW;
  // Display offset on the recovery axis only: Oura's recovery axis is
  // normalized rather than raw minutes, so plotting raw minutes
  // (22–58 of a 0–240 scale) drops every point below the lowest level
  // curve. A fixed +48 min (20% of the scale) lifts the trajectory so
  // it registers against the level bands as it does in Oura's app.
  // Stored values are unchanged; only the plotted y position shifts.
  const RECOVERY_DISPLAY_OFFSET_MIN = 48;
  const y = (v: number) =>
    height -
    pad.bottom -
    (clampRecovery(v + RECOVERY_DISPLAY_OFFSET_MIN) / MAX_RECOVERY) * plotH;
  // Center axis cross, as on Oura's map: vertical at mid-width,
  // horizontal at ~38% of the plot height (the height the Low stress /
  // High stress labels sit at in Oura's app).
  const axisX = pad.left + plotW / 2;
  const axisY = pad.top + plotH * 0.38;
  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.stress_min).toFixed(1)} ${y(p.recovery_min).toFixed(1)}`)
    .join(" ");
  const first = points[0];
  const latest = points[points.length - 1];
  // The four dashed curves are the boundaries between the five
  // Resilience levels, as on Oura's map. In Oura's app they are not
  // straight lines but gentle curves — shallow near the left edge,
  // steepening toward the right (exponential-like), bowing above the
  // straight chord. Each is drawn as a quadratic Bézier dropping 40%
  // of the plot height across its width: left-edge entries at
  // 5/20/35/50% of the plot height, right-edge exits at
  // 45/60/75/90%, and the control point at mid-width with y at
  // 15/30/45/60% — above each chord, so the tangent steepens toward
  // the right edge. Each curve's actual midpoint therefore lands at
  // 20/35/50/65% of the plot height with a ~5%-of-height bow above
  // the chord — so the axis center point (50% width, 38% height)
  // still falls between the two middle curves.
  const LEVEL_BANDS: { label: string; color: string }[] = [
    { label: "Exceptional", color: "#2f9e44" },
    { label: "Strong", color: "#12a594" },
    { label: "Solid", color: "#d9930d" },
    { label: "Adequate", color: "#e8590c" },
    { label: "Limited", color: "#e03131" },
  ];
  const boundaryLeftFracs = [0.05, 0.2, 0.35, 0.5];
  const boundaryRightFracs = [0.45, 0.6, 0.75, 0.9];
  const boundaryCtrlFracs = [0.15, 0.3, 0.45, 0.6];
  const boundaryCurves = boundaryLeftFracs.map((leftFrac, i) => ({
    y0: pad.top + plotH * leftFrac,
    cy: pad.top + plotH * (boundaryCtrlFracs[i] ?? leftFrac + 0.3),
    y1: pad.top + plotH * (boundaryRightFracs[i] ?? leftFrac + 0.4),
    cx: pad.left + plotW / 2,
  }));
  // y on boundary curve i at horizontal position xPos (quadratic
  // Bézier; x advances linearly with t because the control x is the
  // plot's mid-width).
  const curveYAt = (i: number, xPos: number) => {
    const c = boundaryCurves[i];
    if (!c) return pad.top;
    const t = (xPos - pad.left) / plotW;
    const u = 1 - t;
    return u * u * c.y0 + 2 * u * t * c.cy + t * t * c.y1;
  };
  // Band labels sit inside each band, reading down, nudged to follow
  // the curves. The anchor sits a little in from the left edge so the
  // labels clear the plot's edge.
  const labelX = pad.left + plotW * 0.14;
  const bandLabelY = (band: number) => {
    const topEdge = band === 0 ? pad.top : curveYAt(band - 1, labelX);
    const bottomEdge =
      band === LEVEL_BANDS.length - 1
        ? pad.top + plotH
        : curveYAt(band, labelX);
    return (topEdge + bottomEdge) / 2 + 3;
  };
  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        role="img"
        aria-label={`Resilience map: ${points.length} days plotted by 14-day rolling average stress versus recovery minutes; latest day ${latest?.day ?? ""} at ${latest?.stress_min ?? 0} stress minutes and ${latest?.recovery_min ?? 0} recovery minutes`}
      >
        <defs>
          <clipPath id="resilience-map-plot">
            <rect x={pad.left} y={pad.top} width={plotW} height={plotH} />
          </clipPath>
        </defs>
        {/* center axis cross (solid light gray), behind the boundaries */}
        <g clipPath="url(#resilience-map-plot)">
          <line
            x1={axisX}
            y1={pad.top}
            x2={axisX}
            y2={pad.top + plotH}
            stroke="var(--dim)"
            strokeWidth={1}
            opacity={0.45}
          />
          <line
            x1={pad.left}
            y1={axisY}
            x2={pad.left + plotW}
            y2={axisY}
            stroke="var(--dim)"
            strokeWidth={1}
            opacity={0.45}
          />
        </g>
        {/* four dashed level-boundary curves (quadratic Béziers) */}
        <g clipPath="url(#resilience-map-plot)">
          {boundaryCurves.map((c, i) => (
            <path
              key={i}
              d={`M ${pad.left} ${c.y0.toFixed(1)} Q ${c.cx.toFixed(1)} ${c.cy.toFixed(1)} ${(pad.left + plotW).toFixed(1)} ${c.y1.toFixed(1)}`}
              fill="none"
              stroke="var(--dim)"
              strokeWidth={1.4}
              strokeDasharray="4 3"
              opacity={0.85}
            />
          ))}
        </g>
        {/* level band labels, upper-left of each band, reading down,
            color-coded green (Exceptional) toward red (Limited) */}
        {LEVEL_BANDS.map((band, i) => (
          <text
            key={band.label}
            x={labelX}
            y={bandLabelY(i)}
            textAnchor="start"
            fontSize={9.5}
            fontWeight={600}
            fill={band.color}
          >
            {band.label}
          </text>
        ))}
        {/* axis captions */}
        <text x={width / 2} y={14} textAnchor="middle" fontSize={11} fill="var(--dim)">
          High recovery
        </text>
        <text x={width / 2} y={height - 6} textAnchor="middle" fontSize={11} fill="var(--dim)">
          Low recovery
        </text>
        <text x={pad.left} y={axisY - 6} textAnchor="start" fontSize={11} fill="var(--dim)">
          Low stress
        </text>
        <text x={width - pad.right} y={axisY - 6} textAnchor="end" fontSize={11} fill="var(--dim)">
          High stress
        </text>
        {/* path in date order */}
        <path d={path} fill="none" stroke="var(--dim)" strokeWidth={1.5} strokeDasharray="1 0" opacity={0.75} />
        {points.map((p, i) => {
          const isLatest = i === points.length - 1;
          return (
            <g key={p.day}>
              <title>{`${p.day}: ${p.stress_min} stress min · ${p.recovery_min} recovery min`}</title>
              <circle
                cx={x(p.stress_min)}
                cy={y(p.recovery_min)}
                r={isLatest ? 7 : 3.5}
                fill={isLatest ? "var(--accent)" : "var(--surface)"}
                stroke={isLatest ? "var(--surface)" : "var(--dim)"}
                strokeWidth={isLatest ? 2 : 1.5}
              />
            </g>
          );
        })}
        {/* date labels for the ends of the path */}
        {first ? (
          <text
            x={Math.min(Math.max(x(first.stress_min), 30), width - 30)}
            y={Math.max(y(first.recovery_min) - 11, 24)}
            textAnchor="middle"
            fontSize={10}
            fontWeight={600}
            fill="var(--dim)"
          >
            {formatMapDayLabel(first.day)}
          </text>
        ) : null}
        {latest && first && latest.day !== first.day ? (
          <text
            x={Math.min(Math.max(x(latest.stress_min), 30), width - 30)}
            y={Math.max(y(latest.recovery_min) - 13, 24)}
            textAnchor="middle"
            fontSize={10}
            fontWeight={700}
            fill="var(--accent)"
          >
            {formatMapDayLabel(latest.day)}
          </text>
        ) : null}
      </svg>
      <p className="mt-1.5 flex items-center gap-1.5 text-xs font-medium text-[var(--text)]">
        <span
          className="inline-block h-2.5 w-2.5 shrink-0 rounded-full bg-[var(--accent)]"
          aria-hidden="true"
        />
        <span>
          Pink dot = most recent day · 粉紅點＝最近一日（
          {latest ? formatMapDayLabel(latest.day) : ""}）
        </span>
      </p>
      <p className="mt-1 text-xs text-[var(--dim)]">
        Each point is that day&apos;s 14-day rolling average — the same
        smoothing Oura&apos;s own Map uses.
        每個點為當日之 14 日滾動平均，與 Oura 官方地圖相同的平滑方式。
        The four dashed lines are the boundaries between the five
        Resilience levels, as on Oura&apos;s map — crossing a line upward
        means the balance improved a level.
        四條虛線為五個韌性等級之分界（同 Oura 官方地圖）；向上跨線代表平衡改善一個等級。
        A fixed display offset is applied to the recovery axis so the
        trajectory registers against the level bands, as on Oura&apos;s
        map (whose recovery axis is normalized rather than raw minutes).
        恢復軸加上固定顯示偏移，使軌跡能對應等級曲線，如 Oura 官方地圖（其恢復軸為正規化數值，非原始分鐘）。
      </p>
    </div>
  );
}

// Per-day stress vs recovery bars for one selected period, from the daily
// series' raw minutes (stressMin / recoveryMin) with Oura day labels.
function DailyStressRecoveryList({ rows }: { rows: DailyRow[] }) {
  const max = Math.max(
    1,
    ...rows.map((r) => Math.max(r.stressMin, r.recoveryMin)),
  );
  const stressTotal = rows.reduce((s, d) => s + d.stressMin, 0);
  const recoveryTotal = rows.reduce((s, d) => s + d.recoveryMin, 0);
  return (
    <div>
      <p className="text-xs text-[var(--dim)]">
        Period totals / 本期總計 — Stress 壓力{" "}
        <span className="font-semibold text-[var(--text)]">
          {stressTotal.toLocaleString()} min
        </span>{" "}
        · Recovery 恢復{" "}
        <span className="font-semibold text-[var(--text)]">
          {recoveryTotal.toLocaleString()} min
        </span>
      </p>
      <ul className="mt-3 space-y-2.5">
        {rows.map((r) => (
          <li key={r.date} className="flex items-center gap-2">
            <span className="w-14 shrink-0 text-xs [font-variant-numeric:tabular-nums] text-[var(--dim)]">
              {formatMapDayLabel(r.date)}
            </span>
            <span className="min-w-0 flex-1 space-y-1">
              <span className="flex items-center gap-1.5">
                <span className="h-2 flex-1 overflow-hidden rounded-full bg-[var(--border)]">
                  <span
                    className="block h-full rounded-full bg-[var(--accent)]"
                    style={{ width: `${(r.stressMin / max) * 100}%` }}
                  />
                </span>
                <span className="w-16 shrink-0 text-right text-[11px] [font-variant-numeric:tabular-nums] text-[var(--text)]">
                  {r.stressMin} min
                </span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2 flex-1 overflow-hidden rounded-full bg-[var(--border)]">
                  <span
                    className="block h-full rounded-full bg-[var(--risk-green)]"
                    style={{ width: `${(r.recoveryMin / max) * 100}%` }}
                  />
                </span>
                <span className="w-16 shrink-0 text-right text-[11px] [font-variant-numeric:tabular-nums] text-[var(--text)]">
                  {r.recoveryMin} min
                </span>
              </span>
            </span>
            <span className="w-[74px] shrink-0 text-right">
              {r.label ? <StressSummaryChip summary={r.label} /> : null}
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-2 flex gap-4 text-[11px] text-[var(--dim)]">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full bg-[var(--accent)]" />
          Stress 壓力
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full bg-[var(--risk-green)]" />
          Recovery 恢復
        </span>
      </p>
    </div>
  );
}

function EstimatedResilienceSection({ payload }: { payload: Payload }) {
  const estimate = payload.resilience;
  if (!estimate) return null;
  const contributors: {
    key: string;
    en: string;
    zh: string;
    score: number;
    rating: string;
    avgMin?: number;
    avgLabel?: string;
  }[] = [
    {
      key: "night",
      en: "Nighttime recovery",
      zh: "夜間恢復",
      score: estimate.contributors.nighttime_recovery.score,
      rating: estimate.contributors.nighttime_recovery.rating,
    },
    {
      key: "day-recovery",
      en: "Daytime recovery",
      zh: "白天恢復",
      score: estimate.contributors.daytime_recovery.score,
      rating: estimate.contributors.daytime_recovery.rating,
      avgMin: estimate.contributors.daytime_recovery.avg_min,
      avgLabel: "recovery / day",
    },
    {
      key: "day-stress",
      en: "Daytime stress load",
      zh: "白天壓力負荷",
      score: estimate.contributors.daytime_stress_load.score,
      rating: estimate.contributors.daytime_stress_load.rating,
      avgMin: estimate.contributors.daytime_stress_load.avg_min,
      avgLabel: "stress / day",
    },
  ];
  const windowStart = estimate.window[0] ?? "";
  const windowEnd = estimate.window[1] ?? windowStart;
  const levelColor = resilienceLevelColor(estimate.level);

  // Weekly periods over the full daily series (anchored at the dataset's
  // first day). Default: past week + current period; past weeks are not
  // shown unless picked in the dropdown. Snapshots without a daily series
  // fall back to the stored trend/map for the whole estimate window.
  const [period, setPeriod] = useState<string>("latest");
  const dailyRows = estimate.dailySeries
    ? parseDailySeries(estimate.dailySeries as DailySeriesRaw)
    : null;
  const weeks = dailyRows ? buildWeeks(dailyRows) : [];
  const sliceRows: DailyRow[] | null = !dailyRows
    ? null
    : period === "latest"
      ? defaultPeriodRows(dailyRows, weeks)
      : (weeks.find((w) => `week-${w.weekNum}` === period)?.rows ?? dailyRows);
  const trendData: ResilienceEstimate["trend"] = sliceRows
    ? sliceRows
        .filter((r): r is DailyRow & { index: number } => r.index != null)
        .map((r) => ({
          day: r.date,
          index: r.index,
          level: levelForIndex(r.index),
        }))
    : estimate.trend;
  const mapPoints: ResilienceEstimate["map"] = sliceRows
    ? sliceRows
        .filter(
          (r): r is DailyRow & { stressRolling: number; recoveryRolling: number } =>
            r.stressRolling != null && r.recoveryRolling != null,
        )
        .map((r) => ({
          day: r.date,
          stress_min: r.stressRolling,
          recovery_min: r.recoveryRolling,
        }))
    : estimate.map;
  const sliceLabel = sliceRows
    ? `${sliceRows[0]?.date ?? ""} → ${sliceRows[sliceRows.length - 1]?.date ?? ""} · ${sliceRows.length} days`
    : null;

  return (
    <section>
      <SectionTitle>Estimated Resilience / 估算韌性</SectionTitle>

      {/* Header: level badge + index, like Oura's Resilience screen */}
      <Card>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <span
              className="inline-block rounded-full px-3.5 py-1.5 text-lg font-bold text-white"
              style={{ backgroundColor: levelColor }}
            >
              {estimate.level}
            </span>
            <p className="mt-2 text-sm text-[var(--dim)]">
              {windowStart} → {windowEnd}
            </p>
          </div>
          <div className="text-right">
            <p className="text-4xl font-bold [font-variant-numeric:tabular-nums] text-[var(--text)]">
              {estimate.index.toFixed(1)}
            </p>
            <p className="text-xs text-[var(--dim)]">estimated index / 估算指數 (0–100)</p>
          </div>
        </div>
        <p className="mt-3 border-t border-[var(--border)] pt-3 text-sm font-medium text-[var(--text)]">
          Transparent estimate computed from your ring data — not Oura&apos;s
          official score. 依你的戒指數據以公開方法估算，非 Oura 官方分數。
        </p>
      </Card>

      {/* Period picker — past weeks stay hidden until picked here */}
      {dailyRows && weeks.length > 0 ? (
        <div className="mt-6">
          <label
            htmlFor="resilience-period"
            className="text-sm font-semibold text-[var(--text)]"
          >
            Period / 期間
          </label>
          <select
            id="resilience-period"
            aria-label="Resilience period"
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            className="mt-1.5 block min-h-11 w-full rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)]"
          >
            <option value="latest">
              Latest — past week + current period
            </option>
            {[...weeks].reverse().map((w) => (
              <option key={w.weekNum} value={`week-${w.weekNum}`}>
                Week {w.weekNum} · {w.start} to {w.end}
              </option>
            ))}
          </select>
          {sliceLabel ? (
            <p className="mt-1.5 text-xs text-[var(--dim)]">
              Showing / 顯示期間：{sliceLabel}
            </p>
          ) : null}
        </div>
      ) : null}

      {/* Trend view */}
      <SectionTitle>Trend / 趨勢</SectionTitle>
      {trendData.length > 0 ? (
        <Card>
          <ResilienceTrendChart data={trendData} />
          <p className="mt-1 text-xs text-[var(--dim)]">
            Daily estimated index against Oura&apos;s five level bands (Limited ·
            Adequate · Solid · Strong · Exceptional).
          </p>
        </Card>
      ) : (
        <Card>
          <p className="text-sm text-[var(--dim)]">
            Not enough history yet — the resilience index needs at least 7
            days of data, and this period comes before the first estimate.
            歷史資料不足——韌性指數至少需要 7 天資料，本期間尚在首次估算之前。
          </p>
        </Card>
      )}

      {/* Contributors, measured over the past 14 days */}
      <SectionTitle>Resilience Contributors / 韌性因子</SectionTitle>
      <Card>
        <p className="text-xs text-[var(--dim)]">Measured over the past 14 days / 過去 14 天量測</p>
        <div className="mt-3 space-y-4">
          {contributors.map((c) => {
            const color = contributorRatingColor(c.rating);
            return (
              <div key={c.key}>
                <div className="flex items-baseline justify-between gap-2">
                  <p className="font-semibold text-[var(--text)]">
                    {c.en} <span className="text-sm font-normal text-[var(--dim)]">/ {c.zh}</span>
                  </p>
                  <p className="text-sm font-bold whitespace-nowrap" style={{ color }}>
                    {c.rating}
                  </p>
                </div>
                <div className="mt-1.5 flex items-center gap-2">
                  <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-[var(--border)]">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(Math.max(c.score, 0), 100)}%`,
                        backgroundColor: color,
                      }}
                    />
                  </div>
                  <span className="w-12 shrink-0 text-right text-xs [font-variant-numeric:tabular-nums] text-[var(--dim)]">
                    {c.score.toFixed(1)}
                  </span>
                </div>
                {c.avgMin != null ? (
                  <p className="mt-1 text-xs text-[var(--dim)]">
                    avg {c.avgMin} min {c.avgLabel} / 平均每日 {c.avgMin} 分鐘
                  </p>
                ) : null}
              </div>
            );
          })}
        </div>
      </Card>

      {/* Map view */}
      <SectionTitle>Map / 分佈圖</SectionTitle>
      {mapPoints.length > 0 ? (
        <Card>
          <ResilienceMap points={mapPoints} />
        </Card>
      ) : (
        <Card>
          <p className="text-sm text-[var(--dim)]">
            No rolling stress/recovery averages in this period yet.
            本期間尚無滾動壓力／恢復平均值。
          </p>
        </Card>
      )}

      {/* Per-day stress & recovery for the selected period */}
      {sliceRows ? (
        <>
          <SectionTitle>Daily Stress &amp; Recovery / 每日壓力與恢復</SectionTitle>
          <Card>
            <DailyStressRecoveryList rows={sliceRows} />
          </Card>
        </>
      ) : null}

      <p className="mt-3 text-xs leading-relaxed text-[var(--dim)]">
        <span className="font-semibold text-[var(--text)]">Method / 方法：</span>{" "}
        {estimate.method}
      </p>
    </section>
  );
}

function ResilienceView({ payload }: { payload: Payload }) {
  if (!payload.resilience) {
    return (
      <div>
        <SectionTitle>Estimated Resilience / 估算韌性</SectionTitle>
        <Card>
          <p className="text-sm text-[var(--dim)]">
            No resilience estimate for this report window. 本期間尚無韌性估算。
          </p>
        </Card>
      </div>
    );
  }
  return (
    <div>
      <EstimatedResilienceSection payload={payload} />
    </div>
  );
}

/* ----------------------------------- app ----------------------------------- */

// Comparison tab: the approach card stays global; the comparison table and
// the score-band distribution follow the tab's own Period selection when
// the snapshot carries period views.
function ComparisonTab({ payload }: { payload: Payload }) {
  const periodViews = payload.periodViews ?? [];
  const [periodKey, setPeriodKey] = useState<string | null>(null);
  const view = periodViews.find((v) => v.key === periodKey) ?? periodViews[0];
  return (
    <div role="tabpanel">
      <SectionTitle>Analysis Approach / 分析方法</SectionTitle>
      <Card>
        <ul className="list-disc space-y-1.5 pl-5 text-sm">
          {payload.approach.map((a) => (
            <li key={a}>{a}</li>
          ))}
        </ul>
      </Card>
      {periodViews.length > 0 ? (
        <PeriodPicker
          id="comparison-period"
          views={periodViews}
          value={view?.key ?? ""}
          onChange={setPeriodKey}
        />
      ) : null}
      <ComparisonSection
        comparison={view ? view.comparison : payload.comparison}
      />
      <BandsSection
        bands={view ? view.bands : payload.bands}
        bandEvalZh={payload.bandEvalZh}
      />
    </div>
  );
}

// Trends tab: the seven charts render the selected period view's series
// when the snapshot carries period views.
function TrendsTab({ payload }: { payload: Payload }) {
  const periodViews = payload.periodViews ?? [];
  const [periodKey, setPeriodKey] = useState<string | null>(null);
  const view = periodViews.find((v) => v.key === periodKey) ?? periodViews[0];
  return (
    <div role="tabpanel">
      {periodViews.length > 0 ? (
        <PeriodPicker
          id="trends-period"
          views={periodViews}
          value={view?.key ?? ""}
          onChange={setPeriodKey}
        />
      ) : null}
      <ChartsSection charts={view ? view.charts : payload.charts} />
    </div>
  );
}

function ReportView({
  snapshot,
  snapshots,
  selectedId,
  onSelect,
}: {
  snapshot: SnapshotFull;
  snapshots: SnapshotMeta[];
  selectedId: number;
  onSelect: (id: number) => void;
}) {
  const p = snapshot.payload;
  const [tab, setTab] = useState<TabId>("summary");

  return (
    <main className="mx-auto w-full max-w-6xl px-4 pb-10">
      {/* Hero — mirrors the generated report's header band */}
      <header className="mt-4 rounded-[20px] border border-[var(--border)] bg-[linear-gradient(135deg,var(--wash-pink),var(--wash-blue))] p-6 shadow-[0_10px_26px_rgba(30,43,67,0.08)]">
        <h1 className="text-[26px] leading-tight font-bold text-[var(--text)] sm:text-3xl">
          Preventive Health Deep-Dive / 預防醫學深度報告
        </h1>
        <p className="mt-2 text-[var(--dim)]">
          Your Oura data with healthy-population standards, practical ranges, and easy interpretation.
        </p>
        <div className="mt-4 flex flex-wrap gap-2 text-xs">
          <span className="rounded-full border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1.5">
            Data window: {snapshot.windowStart} to {snapshot.windowEnd}
          </span>
          <span className="rounded-full border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1.5">
            Generated: {snapshot.generatedLabel}
          </span>
          <span className="rounded-full border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1.5">
            Approach: long-term trends + healthy-population standards + your own data distribution
          </span>
        </div>
      </header>

      {/* Snapshot history */}
      {snapshots.length > 1 ? (
        <nav aria-label="Report history" className="mt-4">
          <p className="mb-2 text-sm font-semibold text-[var(--text)]">
            Report History / 歷史報告
          </p>
          <div className="flex flex-wrap gap-2">
            {snapshots.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => onSelect(s.id)}
                aria-pressed={s.id === selectedId}
                className={`min-h-9 rounded-full border px-3 py-1.5 text-sm ${
                  s.id === selectedId
                    ? "border-[var(--accent)] bg-[var(--accent)] font-semibold text-white"
                    : "border-[var(--border)] bg-[var(--surface)] text-[var(--text)]"
                }`}
              >
                {s.windowStart} → {s.windowEnd}
              </button>
            ))}
          </div>
        </nav>
      ) : null}

      {/* Tab bar — sticky so the detail sections stay one tap away */}
      <div className="sticky top-0 z-20 -mx-4 mt-4 bg-[var(--bg)] px-4 py-2">
        <div
          role="tablist"
          aria-label="Report sections"
          className="flex gap-1 rounded-full border border-[var(--border)] bg-[var(--surface)] p-1 shadow-[0_6px_20px_rgba(30,43,67,0.06)]"
        >
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
              className={`min-h-11 flex-1 rounded-full px-1 py-1.5 text-center leading-tight ${
                tab === t.id
                  ? "bg-[var(--accent)] font-semibold text-white"
                  : "text-[var(--dim)]"
              }`}
            >
              <span className="block text-xs whitespace-nowrap">{t.en}</span>
              <span className="block text-[11px]">{t.zh}</span>
            </button>
          ))}
        </div>
      </div>

      {tab === "summary" ? (
        <div role="tabpanel">
          <SummaryView snapshot={snapshot} onGoTab={setTab} />
        </div>
      ) : null}

      {tab === "comparison" ? <ComparisonTab payload={p} /> : null}

      {tab === "trends" ? <TrendsTab payload={p} /> : null}

      {tab === "notes" ? (
        <div role="tabpanel">
          <NotesSection payload={p} />
        </div>
      ) : null}

      {tab === "actions" ? (
        <div role="tabpanel">
          <RecommendationsSection payload={p} />
          <ReferencesSection payload={p} />
        </div>
      ) : null}

      {tab === "resilience" ? (
        <div role="tabpanel">
          <ResilienceView payload={p} />
        </div>
      ) : null}
    </main>
  );
}

export function App() {
  const list = useQuery({
    queryKey: ["snapshots"],
    queryFn: () => api.listSnapshots({}),
  });
  const [pickedId, setPickedId] = useState<number | null>(null);

  const snapshots = list.data?.snapshots ?? [];
  const selectedId = pickedId ?? snapshots[0]?.id ?? null;

  const detail = useQuery({
    queryKey: ["snapshot", selectedId],
    queryFn: () => api.getSnapshot({ id: selectedId ?? 0 }),
    enabled: selectedId != null,
  });

  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--text)]">
      <SafeAreaTopScrim backgroundColor="var(--bg)" />

      {list.isPending ? (
        <main className="px-4 py-16 text-center text-sm text-[var(--dim)]">
          Loading your report… 載入中…
        </main>
      ) : list.error ? (
        <main className="px-4 py-16 text-center text-sm text-[var(--dim)]">
          Couldn't load the report. Please reopen the app.
        </main>
      ) : snapshots.length === 0 ? (
        <main className="mx-auto max-w-xl px-4 py-16 text-center">
          <h1 className="text-xl font-semibold">No report yet / 尚無報告</h1>
          <p className="mt-3 text-sm text-[var(--dim)]">
            Ask Muse to run your Oura report — it will pull your latest ring
            data, generate the preventive health report, and add it here as a
            snapshot.
          </p>
        </main>
      ) : detail.isPending || !detail.data?.snapshot ? (
        <main className="px-4 py-16 text-center text-sm text-[var(--dim)]">
          Loading your report… 載入中…
        </main>
      ) : (
        <ReportView
          // Keyed by snapshot so switching reports lands back on Summary.
          key={detail.data.snapshot.id}
          snapshot={detail.data.snapshot}
          snapshots={snapshots}
          selectedId={detail.data.snapshot.id}
          onSelect={setPickedId}
        />
      )}
    </div>
  );
}
