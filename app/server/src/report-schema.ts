// The report payload contract, shared by the actions (request/response
// validation), the seed module, and — through the Actions types — the client.
//
// This mirrors the sections of the generated Oura preventive health report:
// approach bullets, the preventive comparison table, the score band
// distribution, clinical notes, trend chart series, prioritized
// recommendations with a bilingual summary, references, and the disclaimer.
// All display strings are bilingual where the report is (English first,
// Traditional Chinese second), and are carried as data — never hard-coded
// in components — so a refreshed snapshot renders identically in structure.

import { z } from "@hatch/space-sdk";

export const riskTone = z.enum(["red", "yellow", "green"]);
export type RiskTone = z.infer<typeof riskTone>;

export const comparisonRowSchema = z.object({
  metricEn: z.string(),
  metricZh: z.string(),
  risk: riskTone,
  riskLabel: z.string(), // e.g. "High Risk / 高風險"
  unit: z.string(),
  mean: z.string(), // your mean, as displayed (e.g. "7.50")
  spread: z.string(), // e.g. "Median 5.24 (P10-P90: 4.06-6.89)"
  standard: z.string(), // healthy-population standard
  guideline: z.string(), // guideline / target source
  outOfRange: z.string(), // e.g. "0.0% nights < 7 h"
  interpretation: z.string(),
});
export type ComparisonRow = z.infer<typeof comparisonRowSchema>;

export const bandCellSchema = z.object({
  count: z.number(),
  pct: z.number(), // 0-100, one decimal as generated
});
export type BandCell = z.infer<typeof bandCellSchema>;

export const bandRowSchema = z.object({
  nameEn: z.string(),
  nameZh: z.string(),
  optimal: bandCellSchema, // 85-100
  good: bandCellSchema, // 70-84
  fair: bandCellSchema, // 60-69
  attention: bandCellSchema, // 0-59
  total: z.number(),
  noteEn: z.string(),
  noteZh: z.string(),
});
export type BandRow = z.infer<typeof bandRowSchema>;

export const seriesPointSchema = z.object({
  day: z.string(), // YYYY-MM-DD
  value: z.number(),
});
export type SeriesPoint = z.infer<typeof seriesPointSchema>;

export const dualPointSchema = z.object({
  day: z.string(), // YYYY-MM-DD
  a: z.number(),
  b: z.number(),
});
export type DualPoint = z.infer<typeof dualPointSchema>;

export const recommendationSchema = z.object({
  priority: z.number().int().positive(),
  risk: riskTone,
  riskLabel: z.string(), // e.g. "High / 高"
  domainEn: z.string(),
  domainZh: z.string(),
  dataSuggests: z.string(),
  medicalMeaning: z.string(),
  action: z.string(),
  track: z.string(),
});
export type Recommendation = z.infer<typeof recommendationSchema>;

export const stressDaySchema = z.object({
  day: z.string(), // YYYY-MM-DD
  summary: z.string(), // Oura day_summary label, e.g. "stressful"; "" when Oura gave none
});
export type StressDay = z.infer<typeof stressDaySchema>;

// Estimated Resilience — a transparent estimate computed by Muse from the
// user's own pulled ring data (Oura's algorithm is proprietary and its API
// endpoint is not authorized on this connection, so this is NOT Oura's
// official score). Optional: snapshots computed before this field existed
// omit it and render exactly as before.
export const resilienceContributorSchema = z.object({
  score: z.number(), // 0-100
  rating: z.string(), // e.g. "Fair", "Pay attention", "High"
  avg_min: z.number().optional(), // average minutes/day, where applicable
});
export type ResilienceContributor = z.infer<typeof resilienceContributorSchema>;

export const resilienceAnchorSchema = z.object({
  date: z.string(), // YYYY-MM-DD the official reading was captured
  level: z.string(),
  nighttime_recovery: z.string(),
  daytime_recovery: z.string(),
  daytime_stress_load: z.string(),
  source: z.string(),
});
export type ResilienceAnchor = z.infer<typeof resilienceAnchorSchema>;

// One day of the full-dataset daily series behind the resilience estimate:
// [date, indexOrNull, stressRollingOrNull, recoveryRollingOrNull,
//  stressMin, recoveryMin, daySummaryLabel]. Index and rolling values are
// null until enough history exists (the estimate needs at least 7 days);
// the label string "nan" means Oura gave no label and is treated as empty.
export const dailySeriesPointSchema = z.tuple([
  z.string(), // YYYY-MM-DD
  z.number().nullable(), // estimated resilience index for the day
  z.number().nullable(), // 14-day rolling average stress minutes
  z.number().nullable(), // 14-day rolling average recovery minutes
  z.number(), // raw daily stress minutes
  z.number(), // raw daily recovery minutes
  z.string(), // Oura day_summary label; "nan" = none
]);
export type DailySeriesPoint = z.infer<typeof dailySeriesPointSchema>;

export const resilienceEstimateSchema = z.object({
  window: z.array(z.string()), // [start, end] YYYY-MM-DD
  index: z.number(), // 0-100 estimated resilience index
  level: z.string(), // Limited | Adequate | Solid | Strong | Exceptional
  contributors: z.object({
    nighttime_recovery: resilienceContributorSchema,
    daytime_recovery: resilienceContributorSchema,
    daytime_stress_load: resilienceContributorSchema,
  }),
  trend: z.array(
    z.object({
      day: z.string(), // YYYY-MM-DD
      index: z.number(),
      level: z.string(),
    }),
  ),
  map: z.array(
    z.object({
      day: z.string(), // YYYY-MM-DD
      stress_min: z.number(),
      recovery_min: z.number(),
    }),
  ),
  official_anchor: resilienceAnchorSchema,
  // Optional: a second official Oura reading used to calibrate the estimate
  // (v2 estimates carry one; v1 snapshots omit it and render as before).
  official_anchor_2: resilienceAnchorSchema.optional(),
  // Optional: third and fourth official Oura readings (v3 estimates).
  official_anchor_3: resilienceAnchorSchema.optional(),
  official_anchor_4: resilienceAnchorSchema.optional(),
  // Optional: the full-dataset daily series (ALL days, not just the report
  // window). The Resilience tab's weekly-period views render from slices
  // of this; snapshots computed before this field existed omit it and the
  // tab falls back to the stored trend/map above.
  dailySeries: z.array(dailySeriesPointSchema).optional(),
  method: z.string(), // transparent method description, shown as small print
});
export type ResilienceEstimate = z.infer<typeof resilienceEstimateSchema>;

export const chartsSchema = z.object({
  steps: z.array(seriesPointSchema),
  sleepScore: z.array(seriesPointSchema),
  readiness: z.array(seriesPointSchema),
  heartRate: z.array(seriesPointSchema),
  tempDeviation: z.array(seriesPointSchema),
  stressRecovery: z.array(dualPointSchema), // a = stress minutes, b = recovery minutes
  deepRem: z.array(dualPointSchema), // a = deep minutes, b = REM minutes
});
export type Charts = z.infer<typeof chartsSchema>;

// One precomputed period view of the report (e.g. the latest period, all
// data, or one weekly period), in the same shapes the tabs render from the
// top-level payload fields. The Summary, Comparison and Trends tabs each
// offer a Period dropdown over these views when present; snapshots computed
// before this field existed omit it and those tabs render from the
// top-level fields exactly as before.
export const periodViewSchema = z.object({
  key: z.string(), // "latest" | "all" | "week8" | … | "week1"
  labelEn: z.string(),
  labelZh: z.string(),
  start: z.string(), // YYYY-MM-DD, first day of the period
  end: z.string(), // YYYY-MM-DD, last day of the period
  summary: z.object({
    sleepScoreMean: z.number().nullable(), // 0-100, one decimal
    readinessMean: z.number().nullable(), // 0-100, one decimal
    vitals: z.array(
      z.object({
        labelEn: z.string(),
        labelZh: z.string(),
        value: z.string(), // as displayed, unit included (e.g. "5.5 h")
        level: riskTone,
      }),
    ),
    attention: z.array(
      z.object({
        labelEn: z.string(),
        labelZh: z.string(),
        level: riskTone,
      }),
    ),
    stress: z
      .object({
        days: z.number(),
        stressMin: z.number(),
        recoveryMin: z.number(),
        avgStressMin: z.number(),
        avgRecoveryMin: z.number(),
        latestLabel: z.string(), // Oura day_summary of the latest labeled day; "" when none
      })
      .nullable(),
  }),
  comparison: z.array(comparisonRowSchema),
  bands: z.array(bandRowSchema),
  charts: chartsSchema,
});
export type PeriodView = z.infer<typeof periodViewSchema>;

export const reportPayloadSchema = z.object({
  approach: z.array(z.string()),
  comparison: z.array(comparisonRowSchema),
  bands: z.array(bandRowSchema),
  bandEvalZh: z.string(),
  notesEn: z.array(z.string()),
  notesZh: z.array(z.string()),
  charts: chartsSchema,
  // Optional: Oura's per-day stress summary labels (dailystress day_summary).
  // May start one day before charts.stressRecovery; older snapshots omit it.
  stressDays: z.array(stressDaySchema).optional(),
  // Optional: transparent Estimated Resilience computed from the ring data
  // in this snapshot (see resilienceEstimateSchema). Older snapshots omit it.
  resilience: resilienceEstimateSchema.optional(),
  // Optional: precomputed period views (latest / all data / weekly periods)
  // driving the Period dropdowns on the Summary, Comparison and Trends
  // tabs. Older snapshots omit it and those tabs use the top-level fields.
  periodViews: z.array(periodViewSchema).optional(),
  recommendations: z.array(recommendationSchema),
  summaryEn: z.array(z.string()),
  summaryZh: z.array(z.string()),
  priorityNote: z.string(),
  references: z.array(z.string()),
  disclaimer: z.string(),
});
export type ReportPayload = z.infer<typeof reportPayloadSchema>;
